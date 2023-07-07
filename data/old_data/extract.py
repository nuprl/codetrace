from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd

def extract_java(dataset):
   return dataset["train"].filter(lambda example: example["language"] == "java")
   
def extract_java_type_prompts(javads, types = ["int"]):  # ["int", "String"]
   type_prompts = dict(zip(types, [[]]*len(types)))
   for example in javads:
      code = example["prompt"] + example["solution"]
      for typ in types:
         typ_tok = typ + " "
         if typ_tok in code:
            idx_typ = code.index(typ_tok)
            try:
               idx_split = code[idx_typ:].index("=") + len(code[:idx_typ]) + 2
            except:
               continue
            idx_end = code[idx_split:].index(";") + len(code[:idx_split]) +1

            entry = {"prompt": code[:idx_split], # rindex?
                     "solution": code[idx_split : idx_end],
                    }
            type_prompts[typ].append(entry)
   
   return type_prompts
      
def main():
   dataset = load_dataset("nuprl/MultiPL-E-synthetic-solutions") 
   java = extract_java(dataset)
   int_dataset_java = extract_java_type_prompts(java)
   pandas_df = pd.DataFrame(int_dataset_java["int"])
   pandas_df.to_csv("data/int_dataset_java2.csv")
   Dataset.from_pandas(pandas_df).save_to_disk("data/int_dataset_java2")
   
   dataset = load_from_disk("data/int_dataset_java2")
   print(dataset)
   
if __name__=="__main__":
    main()