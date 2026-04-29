import pandas as pd

# clean the binary variables
def clean_binary(x):
    if pd.isna(x):
        return None
    x = str(x).lower().strip()
    
    if x in ["yes", "1", "1.0"]:
        return 1
    if x in ["no", "0", "0.0"]:
        return 0
    
    return None

# clean disclosure
def clean_disclosure(x):
    if pd.isna(x):
        return None
    x = str(x).lower().strip()
    
    if x in ["yes", "1", "1.0", "some of them"]:
        return 1
    if x in ["no", "0", "0.0"]:
        return 0
    
    # drop uncertain responses
    return None

# employer support
def clean_support(x):
    if pd.isna(x):
        return None
    x = str(x).lower()
    
    if "all" in x:
        return 2
    if "some" in x:
        return 1
    if "no" in x or "none" in x:
        return 0
    
    return None

# clean gender
def clean_gender(x):
    if pd.isna(x):
        return None
    
    x = str(x).lower().strip()
    
    # female first (so "female" doesn't get caught as "male")
    if any(word in x for word in ["female", "f", "woman", "cis-female"]):
        return "female"
    
    if any(word in x for word in ["male", "m", "man", "cis-male"]):
        return "male"
    
    # everything else
    return "other"