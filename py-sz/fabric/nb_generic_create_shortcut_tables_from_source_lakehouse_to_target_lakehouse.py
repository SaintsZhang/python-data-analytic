#!/usr/bin/env python
# coding: utf-8

# ## nb_generic_create_shortcut_tables_from_source_lakehouse_to_target_lakehouse
# 
# New notebook

# In[1]:


import sempy.fabric as fabric
import sempy_labs as labs
import json


# In[2]:


p_source_table_names = ""
p_source_lakehouse_name = ""
p_source_workspace_name = ""
p_target_lakehouse_name = ""
p_target_workspace_name = ""


# In[3]:


# p_source_table_names = """
# {
#   "tables": [
#     { "source_table": "fivetran_metadata__connection", "shortcut_name": "fivetran_metadata__connection_raw" },
#     { "source_table": "fivetran_metadata__destination", "shortcut_name": "fivetran_metadata__destination_raw" }
#   ]
# }
# """
# p_source_lakehouse_name = "lh_fivetran_sap"
# p_source_workspace_name = "Data Platform Setup Bronze | Sandbox"
# p_target_lakehouse_name = "lh_data_ocean_curated_sap_ecc"
# p_target_workspace_name = "Data Platform Setup Bronze | Sandbox"


# In[4]:


cfg = json.loads(p_source_table_names)
results = []


# In[5]:


for item in cfg["tables"]:
    source_table = item["source_table"]
    shortcut_name = item.get("shortcut_name")
    try:
        labs.lakehouse.create_shortcut_onelake(
            source_table,
            p_source_lakehouse_name,
            p_source_workspace_name,
            p_target_lakehouse_name,
            p_target_workspace_name,
            shortcut_name
        )
        results.append({"source_table": source_table, "shortcut_name": shortcut_name, "status": "created"})
    except Exception as e:
        results.append({"source_table": source_table, "shortcut_name": shortcut_name, "status": "failed", "error": str(e)})


# In[6]:


results

