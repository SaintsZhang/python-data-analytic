#!/usr/bin/env python
# coding: utf-8

# ## nb_generic_create_shortcut_table_from_source_lakehouse_to_target_lakehouse
# 
# null

# In[2]:


import sempy.fabric as fabric
import sempy_labs as labs


# In[3]:


p_source_table_name = ""
p_source_lakehouse_name = ""
p_source_workspace_name = ""
p_target_lakehouse_name = ""
p_target_workspace_name = ""
p_shortcut_name = ""


# In[4]:


# p_source_table_name = "fivetran_metadata__user"
# p_source_lakehouse_name = "lh_fivetran_sap"
# p_source_workspace_name = "Data Platform Setup Bronze | Sandbox"
# p_target_lakehouse_name = "lh_data_ocean_curated_sap_ecc"
# p_target_workspace_name = "Data Platform Setup Bronze | Sandbox"
# p_shortcut_name = "fivetran_metadata__user_raw"


# In[5]:


labs.lakehouse.create_shortcut_onelake(p_source_table_name,p_source_lakehouse_name,p_source_workspace_name,p_target_lakehouse_name,p_target_workspace_name,p_shortcut_name)

