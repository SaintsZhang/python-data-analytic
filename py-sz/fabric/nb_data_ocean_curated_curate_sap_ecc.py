#!/usr/bin/env python
# coding: utf-8

# ## nb_data_ocean_curated_curate_sap_ecc
# 
# null

# In[1]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %%configure -f
# {
#     "defaultLakehouse": 
#     {
#         "name" : { 
#             "parameterName": "defaultLakehouseName", 
#             "defaultValue": "lh_data_ocean_curated_sap_ecc" 
#         },
#         "workspaceId": { 
#             "parameterName": "workspaceId", 
#             "defaultValue": ""
#         }
#     }
# }


# In[2]:


import sempy.fabric as fabric
from pyspark.sql.functions import lit, col, round,year, current_date, col
from pyspark.sql import functions as F


# In[3]:


p_target_table_name = ""
p_load_datetime = ""


# In[ ]:


# p_target_table_name = "fivetran_metadata__user"
# p_load_datetime = "20251223"


# In[4]:


workspace_id = fabric.get_workspace_id()
lakehouse_id = fabric.get_lakehouse_id()
lakehouse_path = f"abfss://{workspace_id}@onelake.dfs.fabric.microsoft.com/{lakehouse_id}"
table_path = f"{lakehouse_path}/Tables/{p_target_table_name}_raw"


# In[5]:


df = spark.read.format("delta").load(table_path)


# In[6]:


df = df.withColumn("fabric_load_datetime", lit(p_load_datetime))


# In[7]:


cols = df.columns


# In[8]:


df = df.withColumn(
    "row_hash",
    F.sha2(
        F.concat_ws(
            "||",
            *[F.coalesce(F.col(c).cast("string"), F.lit("")) for c in cols]
        ),
        256
    )
)


# In[10]:


df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(p_target_table_name)

