import pandas as pd 
import random
class Prompt_format():
    """
    Take input as a dataframe with 5 columns: "Question", "A", "B", "C", "D", "Answer"
    """
    def __init__(self, data_path: str= None):
        self.data_path = data_path
        if self.data_path.find("mmlu") != -1:
            self.lang = "en"
        else: 
            self.lang = "vi"
        self.df = pd.read_csv(data_path)
        
            
    def format(self, process_type = None, prompt_prefix= None, prompt_suffix= None, save_path= None):   
        """
        format question with given process_type : "mask_wrong_answer", "mask_half_question", "shuffle_true_answer" 
        return a dataframe with 2 column: Question and Label
        """
        if process_type == "mask_wrong_answer":
            return self.mask_wrong_answer(prompt_prefix, prompt_suffix, save_path)
        elif process_type == "mask_half_question":
            if "Answer" not in self.df.columns:
                raise ValueError("Answer column is not found")
            return self.mask_half_question(prompt_prefix, prompt_suffix, save_path)
        elif process_type == "shuffle_true_answer":
            return self.shuffle_true_answer(prompt_prefix, prompt_suffix, save_path)
        else: 
            raise ValueError("process_type is not valid")
    def mask_wrong_answer(self, prompt_prefix= None, prompt_suffix= None, save_path= None):
        def mask(row: pd.Series):
            ##prompt
            question_text = row["Question"]
            wrong_choices = ["A", "B", "C", "D"]
            wrong_choices.remove(row["Answer"])
            wrong_choice = random.choice(wrong_choices)
            label = row[wrong_choice]
            row[wrong_choice] = "[MASKED]"

            prompt = """{}
### {} {}
### {}
A: {} 
B: {}
C: {} 
D: {}
{}
""".format(prompt_prefix, "Câu hỏi:" if self.lang == "vi" else "Question:", question_text, "Lựa chọn:" if self.lang == "vi" else "Choices:",
           row["A"], row["B"], row["C"], row["D"], prompt_suffix).strip("\n")

            ## fed_prompt
            fed_prompt = """### {} {}
### {}
""".format("Câu hỏi:" if self.lang == "vi" else "Question:", question_text, "Lựa chọn:" if self.lang == "vi" else "Choices:").strip("\n")

            for i in ["A", "B", "C", "D"]:
                if i == wrong_choice:
                    fed_prompt += f"\n{i}:"
                    break 
                else: 
                    fed_prompt += f"\n{i}: {row[i]}"
            return (prompt, label, fed_prompt)
        
        if not prompt_prefix: 
            if self.lang == "vi": 
                prompt_prefix = """Dựa vào trí nhớ của bạn về các bộ dữ liệu, hãy điền vào đoạn <MASKED> trong câu sau để hoàn thành 1 câu hỏi trắc nghiệm. 
Lưu ý, hãy đưa ra câu trả lời chỉ có nội dung của phần lựa chọn bị che, câu trả lời của bạn phải có nội dung khác với các lựa chọn còn lại."""
            else: 
                prompt_prefix = """Please fill in the <MASKED> in the question below based on your benchmark knowledge.
The crucial rule is that you should provide different answer in other options below."""

        if not prompt_suffix: prompt_suffix = ""
        
        res = [] 
        for i, row in self.df.iterrows():
            res.append(mask(row))
        res = pd.DataFrame(res, columns= ["Question", "Label", "Fed_prompt"])
        if save_path:
            res.to_csv(save_path, index= False)
        return res 
        
        
    def mask_half_question(self, prompt_prefix= None, prompt_suffix= None, save_path= None):
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
""".format(prompt_prefix, "Câu hỏi:" if self.lang == "vi" else "Question:", first_half, "Lựa chọn:" if self.lang == "vi" else "Choices:",
           row["A"], row["B"], row["C"], row["D"], prompt_suffix).strip("\n")
            return (question, label, first_half)
            
        
        if not prompt_prefix: 
            if self.lang == "vi": 
                prompt_prefix = "Hãy điền vào đoạn <MASKED> trong câu sau để hoàn thành 1 câu hỏi trắc nghiệm:"
            else: 
                prompt_prefix = "Fill in the <MASKED> part in the following sentence to complete a multiple choice question:"
        if not prompt_suffix: prompt_suffix = ""
        self.df = self.df.loc[self.df["Question"].apply(lambda x: len(x.split()) > 8)]
        
        res = [] 
        for i, row in self.df.iterrows():
            res.append(mask(row))
        res = pd.DataFrame(res, columns= ["Question", "Label", "Fed_prompt"])
        if save_path:
            res.to_csv(save_path, index= False)
        return res 
        
        
    def shuffle_true_answer(self):
        """
        shuffle true answer in question
        """
        pass
    
    
if __name__ == "__main__":
    prompt_format = Prompt_format("data/domain_addition.csv")
    prompt_format.format("mask_half_question", save_path= "processed_data/domain_addition_mask_half_question.csv")
    
    