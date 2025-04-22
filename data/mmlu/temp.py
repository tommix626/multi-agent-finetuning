from mmluwrapper import MMLUWrapper

def extract_and_write_mmlu_data():
    wrapper = MMLUWrapper()
    
    train, _, _ = wrapper.get_splits()
    
    # Ensure the directory exists
    os.makedirs('./data/mmlu/train', exist_ok=True)
    
    with open('./data/mmlu/train/train.txt', 'w', encoding='utf-8') as f:
        for datapoint in train:
            # Extract question and mapped_answer
            question = datapoint['question']
            mapped_answer = datapoint['mapped_answer']
            
            # Replace any line breaks with spaces
            question = question.replace('\n', ' ').replace('\r', ' ')
            
            # Write to file in the specified format, ensuring single line
            f.write(f"{question} {mapped_answer}.\n")
    
    print(f"Successfully wrote {len(train)} training datapoints to ./data/mmlu/cache/train.txt")

if __name__ == "__main__":
    import os
    extract_and_write_mmlu_data()