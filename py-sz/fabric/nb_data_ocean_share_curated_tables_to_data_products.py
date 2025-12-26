#!/usr/bin/env python
# coding: utf-8

# ## nb_data_ocean_share_curated_tables_to_data_products
# 
# null

# In[1]:


import json
from typing import List

def build_tables_json(tables: List[str], suffix: str) -> str:
    # Normalize suffix: allow "raw" or "_raw"
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    payload = {
        "tables": [
            {"source_table": t, "shortcut_name": f"{t}{suffix}"}
            for t in tables
        ]
    }
    # Pretty JSON (use separators=(",", ":") if you want it minified)
    return json.dumps(payload, indent=2)


# In[2]:


tables = ["fivetran_metadata__connection", "fivetran_metadata__destination"]
suffix = "_raw"

tables_json = build_tables_json(tables, suffix)
print(tables_json)


# In[3]:


from typing import Dict, Any

args: Dict[str, Any] = {
    "p_source_table_names": tables_json,
    "p_source_lakehouse_name": "lh_fivetran_sap",
    "p_source_workspace_name": "Data Platform Setup Bronze | Sandbox",
    "p_target_lakehouse_name": "lh_data_ocean_curated_sap_ecc",
    "p_target_workspace_name": "Data Platform Setup Bronze | Sandbox"
}


# In[4]:


result = mssparkutils.notebook.run('nb_generic_create_shortcut_tables_from_source_lakehouse_to_target_lakehouse', 90, args)


# In[5]:


result

