import pandas as pd 
import ast

class MMLUProcessor():
    
    def __init__(self, data_path= "raw_data/mmlu_test.csv", size= None):
        self.df = pd.read_csv(data_path)
        if size: 
            self.df = self.df.sample(size)
        self.df["choices"] = self.df["choices"].apply(ast.literal_eval)
        
    def process(self, save_path= "data/mmlu.csv"):
        """
        process mmlu data 
        return: 
        dataframe with 5-6 columns: "Question", "A", "B", "C", "D", "Answer"
        """
        self.df.rename(columns= {"question": "Question"}, inplace= True)
        self.df["A"] = self.df["choices"].apply(lambda x: x[0]).astype(str)
        self.df["B"] = self.df["choices"].apply(lambda x: x[1]).astype(str)
        self.df["C"] = self.df["choices"].apply(lambda x: x[2]).astype(str)
        self.df["D"] = self.df["choices"].apply(lambda x: x[3]).astype(str)
        idx_to_answer = {0: "A", 1: "B", 2: "C", 3: "D"}
        self.df["Answer"] = self.df["answer"].apply(lambda x: idx_to_answer[x])
        self.df.drop(columns= ["choices", "answer", "subject"], inplace= True)
        if save_path:
            self.df.to_csv(save_path, index= False)
        
        return self.df
    
if __name__ == "__main__":
    mmlu_processor = MMLUProcessor()
    mmlu_processor.process()