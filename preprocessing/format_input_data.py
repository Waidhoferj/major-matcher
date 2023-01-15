
import pandas as pd
from nltk.tokenize import sent_tokenize
from build_data_dict import build_course_program_dict

OUTPUT_FILE = "course_sentences.csv"

invalid_phrases = [line.rstrip('\n') for line in open(
        'stopwords/invalid_description_phrases.txt')]  # Load .txt file line by line

def is_valid_sentence(sentence):
    if sentence == "":
        return False;
    return all(phrase not in sentence.lower() for phrase in invalid_phrases)

if __name__ == "__main__":
    courses_df = pd.read_csv("courses.csv")
    course_program_dict = build_course_program_dict()

    rows = []
    for course, programs in course_program_dict.items():
        # only capture unique courses
        if (len(programs) > 1):
            continue

        course_row = courses_df.loc[courses_df['Course Prefix'] == course]
        
        if(len(course_row["Description"].values) == 0):
            continue;

        course_description = course_row["Description"].values[0]
        sentences = sent_tokenize(course_description)
        sentences = [sentence.strip() for sentence in sentences if is_valid_sentence(sentence)]
        

        # if a course belongs to more than one program, use the department as the program
        if len(programs) > 1:
            dept = course_row["Dept"].values[0]
            for sentence in sentences:
                rows.append([sentence, course, dept])
            continue
        else:
            for program in programs:
                for sentence in sentences:
                    rows.append([sentence, course, program])

    output_df = pd.DataFrame(rows, columns=["sentence", "course", "program"])
    output_df.to_csv(OUTPUT_FILE, index=False)