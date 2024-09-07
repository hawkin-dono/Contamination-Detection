import pandas as pd 
import random
class Prompt_format():
    """
    Take input as a dataframe with 5 columns: "Question", "A", "B", "C", "D", "Answer"
    """
    def __init__(self, data_path: str= None):
        self.data_path = data_path
        if self.data_path.startswith("mmlu"):
            self.lang = "en"
        else: 
            self.lang = "vi"
            
    def format(self, type = None, prompt_prefix= None, prompt_suffix= None, save_path= None):   
        """
        format question with given type : "mask_wrong_answer", "mask_half_question", "shuffle_true_answer" 
        return a dataframe with 2 column: Question and Label
        """
        df = pd.read_csv(self.data_path)
        if type == "mask_wrong_answer":
            return self.mask_wrong_answer(df, prompt_prefix, prompt_suffix, save_path)
        elif type == "mask_half_question":
            return self.mask_half_question(df, prompt_prefix, prompt_suffix, save_path)
        elif type == "shuffle_true_answer":
            return self.shuffle_true_answer(df, prompt_prefix, prompt_suffix, save_path)
        else: 
            raise ValueError("Type is not valid")
    def mask_wrong_answer(self, df: pd.DataFrame, prompt_prefix= None, prompt_suffix= None, save_path= None):
        pass
    def mask_half_question(self, df: pd.DataFrame, prompt_prefix= None, prompt_suffix= None, save_path= None):
        """
        mask half question
        """
        def mask(row: pd.Series):
            text = row["Question"]
            words = text.split()
            first_half = words[:len(words)//2]
            second_half = words[len(words)//2:] 
             
            first_half = " ".join(first_half) 
            label = " ".join(second_half)
            question = """{}
### {} {} [MASKED]
### {}
A: {} 
B: {}
C: {} 
D: {}
{}
""".format(prompt_prefix, "Câu hỏi" if self.lang == "vi" else "Question", first_half, "Lựa chọn" if self.lang == "vi" else "Choices",
           row["A"], row["B"], row["C"], row["D"], prompt_suffix).strip("\n")
            return (question, first_half, label)
            
        
        if not prompt_prefix: 
            if self.lang == "vi": 
                prompt_prefix = "Hãy điền vào đoạn <MASKED> trong câu sau để hoàn thành 1 câu hỏi trắc nghiệm: \n"
            else: 
                prompt_prefix = "Fill in the <MASKED> part in the following sentence to complete a multiple choice question: \n"
        if not prompt_suffix: prompt_suffix = ""
        df = df.loc[df["Question"].apply(lambda x: len(x.split()) > 8)]
        
        res = [] 
        for i, row in df.iterrows():
            res.append(mask(row))
        res = pd.DataFrame(res, columns= ["Question", "First_half", "Label"])
        if save_path:
            res.to_csv(save_path, index= False)
        return res 
        
        
    def shuffle_true_answer(self):
        """
        shuffle true answer in question
        """
        pass
    
    
if __name__ == "__main__":
    prompt_format = Prompt_format("data/mmlu.csv")
    prompt_format.format("mask_half_question", save_path= "data/mmlu_mask_half_question.csv")
    
    