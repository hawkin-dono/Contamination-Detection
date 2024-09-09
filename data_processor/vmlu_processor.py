import pandas as pd 
import ast

class VMLUProcessor():
    
    def __init__(self, data_path= "raw_data/VMLU_test.jsonl", size: int= None):
        
        self.df = pd.read_json(data_path, lines= True)
        if size:
            self.df = self.df.sample(size)
        
    def process(self, save_path= "data/vmlu.csv"):
        """
        process mmlu data 
        return: 
        dataframe with 5-6 columns: "Question", "A", "B", "C", "D", "Answer"
        """
        self.df.rename(columns= {"question": "Question"}, inplace= True)
        self.df["A"] = self.df["choices"].apply(lambda x: str(x[0])[3:])
        self.df["B"] = self.df["choices"].apply(lambda x: str(x[1])[3:])
        self.df["C"] = self.df["choices"].apply(lambda x: str(x[2])[3:])
        self.df["D"] = self.df["choices"].apply(lambda x: str(x[3])[3:] if len(x) == 4 else "")
        self.df.drop(columns= ["id", "choices"], inplace= True)
        if save_path:
            self.df.to_csv(save_path, index= False)
        return self.df
    
if __name__ == "__main__":
    mmlu_processor = VMLUProcessor()
    mmlu_processor.process()