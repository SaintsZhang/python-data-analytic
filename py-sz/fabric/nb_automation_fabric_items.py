#!/usr/bin/env python
# coding: utf-8

# ## nb_automation_fabric_items
# 
# null

# ### Workspace and items creation notebook

# In[2]:


# Parameters that will be overwriten by pipeline notebook calls
env = "dev"
layer = "data_ocean"
scope = "raw"
source = "sap_ecc"
admin_group_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxx"
contrib_group_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxx"
view_group_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxx"
# Lists of items to create, by purposes
pipelines = ["data_ingest","complex_transform"]
notebooks = ["training","explore"]
dataflows = ["tranform_stuff"]


# ### Plan

# In[ ]:


# TODO GC : from all inputs definine the list of resources to create, based on layer and scope


# ### Input validation

# In[ ]:


# Validate inputs
envs = ["dev","tst","prd"]
layers = ["data_ocean","data_product"]
scopes = ["raw","curated","conform"]
assert env in envs, f"Invalid env: {env}. Expected one of {envs}"
assert layer in layers, f"Invalid layer: {layer}. Expected one of {layers}"
assert scope in scopes, f"Invalid scope: {scope}. Expected one of {scopes}"


# In[ ]:


# List usable capacities, set capacity_id to use
capacities = fabric.list_capacities()
print(capacities)
capacity_id = "87d37f65-e350-4762-b3e0-1713eaf7a8ca" # Premium Per User - Reserved  PP3 - North Europe


# ### Workspace creation

# In[ ]:


# Create workspace if not exist 
# asigning capa at creation
workspace_name = f"ws_{layer}_{scope}_{source}_{env}"
try:
    workspaces = fabric.list_workspaces()
    
    if workspace_name not in workspaces['Name'].values:
        print(f"Creating workspace: {workspace_name}...")
        fabric.create_workspace(workspace_name,capacity_id,workspace_name)
        print("Workspace created successfully.")
    else:
        print(f"Workspace '{workspace_name}' already exists.")
        
except Exception as e:
    print(f"Failed to manage workspace: {e}")
    raise


# > ⚠️ Due to race condition, the group access asignment might need notebook re-authentication (new spark session), to be able to grant access to a workspace freshly create.

# In[ ]:


# Grant access to workspace from groups
import requests
import notebookutils
import sempy.fabric as fabric
def assign_workspace_role(workspace_name, principal_id, role="Viewer", principal_type="Group"):
    """
    Assigns a role to a principal in a Fabric workspace using the native v1 API.
    
    Args:
        workspace_name (str): The name of the workspace.
        principal_id (str): The Entra ID (Object ID) of the User or Group.
        role (str): Admin, Member, Contributor, or Viewer.
        principal_type (str): User, Group, or ServicePrincipal.
    """

    fabric_resource = "https://api.fabric.microsoft.com/"
    token = notebookutils.credentials.getToken(fabric_resource)
    workspace_id = fabric.resolve_workspace_id(workspace_name)

    url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/roleAssignments"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "principal": {
            "id": principal_id,
            "type": "Group" 
        },
        "role": role
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code in [200, 201]:
        print(f"Success: Assigned {role} role to principal {principal_id}")
    else:
        print(f"Failed: {response.status_code} - {response.text}")

assign_workspace_role(workspace_name,admin_group_id,"Admin","Group")
assign_workspace_role(workspace_name,contrib_group_id,"Contributor","Group")
assign_workspace_role(workspace_name,view_group_id,"Viewer","Group")


# ## Lakehouse and warehouse creation

# In[ ]:


import requests
import sempy.fabric as fabric
import notebookutils 
lakehouse_name = f"lh_{layer}_{scope}_{source}"
warehouse_name = f"wh_{layer}_{scope}_{source}"
try:
    #Lakehouse Creation
    lakehouses = fabric.list_items(type="Lakehouse",workspace=workspace_name)
    existing_lakehouses = lakehouses['Display Name'].values if not lakehouses.empty else []


    if lakehouse_name not in existing_lakehouses:
        print(f"Creating Lakehouse: {lakehouse_name}...")
        fabric.create_lakehouse(display_name=lakehouse_name,description=lakehouse_name,max_attempts=10,workspace=workspace_name)
    else:
        print(f"Lakehouse '{lakehouse_name}' already exists.")

    #Warehouse Creation
    warehouses = fabric.list_items("Warehouse",workspace_name)
    existing_warehouses = warehouses['Display Name'].values if not warehouses.empty else []

    if warehouse_name not in existing_warehouses:
        print(f"Creating Warehouse: {warehouse_name}...")

        fabric_resource = "https://api.fabric.microsoft.com/"
        token = notebookutils.credentials.getToken(fabric_resource)
        workspace_id = fabric.resolve_workspace_id(workspace_name)

        url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/warehouses"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "displayName": warehouse_name
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("Warehouse creation initiated.")
    else:
        print(f"Warehouse '{warehouse_name}' already exists.")

except Exception as e:
    print(f"Automation failed: {e}")
    raise


# ## Notebooks, pipelines and dataflows creation

# In[ ]:


import requests
import sempy.fabric as fabric
import notebookutils 

item_configs = {
    "DataPipeline": {"prefix": f"pl_{layer}_{scope}", "items": pipelines, "endpoint": "dataPipelines"},
    "Notebook":     {"prefix": f"nb_{layer}_{scope}", "items": notebooks, "endpoint": "notebooks"},
    "Dataflow":     {"prefix": f"df_{layer}_{scope}", "items": dataflows, "endpoint": "dataflows"}
}

try:
    workspace_id = fabric.resolve_workspace_id(workspace_name)
    token = notebookutils.credentials.getToken("https://api.fabric.microsoft.com/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    all_existing = fabric.list_items(workspace=workspace_name)

    for item_type, config in item_configs.items():
        existing_names = all_existing[all_existing['Type'] == item_type]['Display Name'].values if not all_existing.empty else []
        
        for purpose in config["items"]:
            display_name = f"{config['prefix']}_{purpose}"
            
            if display_name not in existing_names:
                print(f"Creating {item_type}: {display_name}...")
                url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/{config['endpoint']}"
                
                response = requests.post(url, headers=headers, json={"displayName": display_name})
                response.raise_for_status()
                print(f"{item_type} created.")
            else:
                print(f"{item_type} '{display_name}' already exists.")

except Exception as e:
    print(f"Automation failed: {e}")
    raise

