import pandas as pd
 
class DomainProcessor:
    def __init__(self, data_path= "raw_data/domain.csv", size: int= None):

        self.df = pd.read_csv(data_path)
        if size:
            self.df = self.df.sample(size)
    
    def process(self, save_path= "data/domain.csv"):
        """
        process domain data 
        return: 
        dataframe with 5-6 columns: "Question", "A", "B", "C", "D", "Answer"
        """
        self.df["Question"] = self.df["Question"].astype(str)
        self.df["A"] = self.df["A"].astype(str)
        self.df["B"] = self.df["B"].astype(str)
        self.df["C"] = self.df["C"].astype(str)
        self.df["D"] = self.df["D"].astype(str)
        self.df["Answer"] = self.df["Answer"].astype(str)
        self.df.drop(columns= ["subject"], inplace= True)
        
        if save_path:
            self.df.to_csv(save_path, index= False)
        
        return self.df

if __name__ == "__main__":
    domain_processor = DomainProcessor()
    domain_processor.process()
        
        
        