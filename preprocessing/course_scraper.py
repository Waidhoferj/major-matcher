import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Tuple

FILE_NAME: str = "courses.csv"
MATRIX_FILE_NAME: str = "catalog_matrix.csv"

BASE_COURSE_CATALOG_URL: str = "https://catalog.calpoly.edu/coursesaz/"
BASE_COLLEGES_DEPT_URL: str = "https://catalog.calpoly.edu/coursesaz/#courseprefixestext"


def getTextWithinParentheses(text: str):
    return text[text.find('(')+1:text.find(')')]


def clean_link_text(text):
    return text.replace(')', "").replace('(', "").strip()


def parse_college_html(college_html) -> List[Tuple[str, List[str]]]:
    depts = []
    college_dept_pairs = []
    current_college = None
    for child in college_html.children:
        tag = child.name
        text = child.text
        # dept
        if(tag == 'a'):
            depts.append(clean_link_text(text))
        # college
        elif(tag == 'strong'):
            if(current_college is not None):
                college_dept_pairs.append([current_college, depts])
                depts = []
            current_college = text.strip()
    college_dept_pairs.append((current_college, depts))
    return college_dept_pairs


def parse_college_department_html(html) -> List[str]:
    depts = []
    for dept_html in html.children:
        if(dept_html.name is not None):
            dept = dept_html.text.split('(')[0].strip()
            prefixes = [a.text.replace(')', "")
                        for a in dept_html.find_all("a")]
            depts.append((dept, prefixes))
    return depts


# iterates through all depts of a college
# {
#     'Biological Science': [BIO, BOT MCRO, MSCI],
#     'Chemistry and Biochemisty': [CHEM]
#       ...
# }
def create_depts_dict(depts):
    d = {}
    for dept in depts:
        d |= {
            dept[0]: dept[1]
        }
    return d


def scrape_course_prefixes():
    DEPT_PREFIXES_ID: str = "courseprefixestextcontainer"
    course_page = requests.get(BASE_COLLEGES_DEPT_URL)
    soup = soup = BeautifulSoup(course_page.content, "html.parser")

    colleges_html = soup.find(id=DEPT_PREFIXES_ID).div
    college_dept_dict = {}
    current_college = None
    depts = []

    for child in colleges_html.children:
        tag = child.name
        if (tag == "ul"):  # nested departments
            depts = parse_college_department_html(child)
        elif (tag == 'p'):  # college
            college_dept_pairs = parse_college_html(child)
            # more than one, so add all but the last to dictionary
            if(len(college_dept_pairs) > 1):
                # add current college and departments
                college_dept_dict |= {
                    current_college[0]: {
                        current_college[0]: current_college[1]
                    }
                }
                college_dept_dict[current_college[0]
                                  ] |= create_depts_dict(depts)
                # add all other colleges except for the last one
                for i in range(len(college_dept_pairs)-1):
                    # add colleges with no departments
                    college_dept_dict |= {
                        college_dept_pairs[i][0]: {
                            college_dept_pairs[i][0]: college_dept_pairs[i][1]
                        }
                    }
                # set current college to last in pairs
                current_college = college_dept_pairs[-1]
            else:
                if (current_college is not None):
                    # add current college and departments
                    college_dept_dict |= {
                        current_college[0]: {
                            current_college[0]: current_college[1]
                        }
                    }
                    # add college departments
                    college_dept_dict[current_college[0]
                                      ] |= create_depts_dict(depts)
                # set new current college
                current_college = college_dept_pairs[0]

    # add last current college
    college_dept_dict |= {
        current_college[0]: {
            current_college[0]: current_college[1]
        }
    }
    # add college departments
    college_dept_dict[current_college[0]
                      ] |= create_depts_dict(depts)

    return college_dept_dict


def extract_course_info(data, college, dept, prefix):
    prefix = prefix.lower()
    url = f'{BASE_COURSE_CATALOG_URL}/{prefix}'
    page = requests.get(url)

    # scrape data
    soup = BeautifulSoup(page.content, "html.parser")
    courses = soup.find_all("div", class_="courseblock")

    if (college == dept):
        dept = f'{dept} Dept'

    for c in courses:
        course_name: List[str] = c.find(
            "p", class_="courseblocktitle").strong.contents[0].split(".")
        course_num: str = course_name[0].replace(
            "\xa0", "-").strip()  # replace nonbreaking space
        name: str = course_name[1].strip()
        units: str = c.find("span", class_="courseblockhours").text.strip()
        description: str = c.find(
            "div", class_="courseblockdesc").p.text.strip()
        data.append([college, dept, course_num,
                    name, units, description, college+dept, dept+course_num])


def scrape_courses(prefixes_dict):
    data = []
    for college in prefixes_dict.keys():
        for dept in prefixes_dict[college]:
            for prefix in prefixes_dict[college][dept]:
                if(prefix):
                    extract_course_info(data, college, dept, prefix)
    return data


def build_df(data):
    column_names = ["College", "Dept", "Course Prefix",
                    "Course Name", "Units", "Description", "College+Dept", "Dept+CourseNum"]
    df = pd.DataFrame(data, columns=column_names)
    return df


def find_match(course_list, matrix):
    for row in matrix.index:
        for col in matrix.columns:
            match = not(course_list[(course_list['College+Dept'] == row+col)
                                    ].empty) or not(course_list[(course_list['Dept+CourseNum'] == row+col)].empty)
            if(match):
                matrix.loc[row, col] = 1
                # print(row, col)
    return


def build_adj_matrix(course_list: pd.DataFrame):
    colleges = list(course_list["College"].unique())
    depts = list(course_list["Dept"].unique())
    courses = list(course_list["Course Prefix"].unique())

    indices = [(1, college) for college in colleges] + \
        [(2, dept) for dept in depts] + [(3, course) for course in courses]

    multi_index = pd.MultiIndex.from_tuples(indices)
    adj_matrix = pd.DataFrame(index=multi_index, columns=multi_index).fillna(0)

    # grab necessary sections
    colleges_to_depts = adj_matrix.loc[1, 2]
    depts_to_courses = adj_matrix.loc[2, 3]

    # mark matches
    find_match(course_list, colleges_to_depts)
    find_match(course_list, depts_to_courses)
    return adj_matrix


if __name__ == "__main__":
    prefixes_dict = scrape_course_prefixes()
    courses = scrape_courses(prefixes_dict)
    course_list = build_df(courses)
    course_list.to_csv(FILE_NAME, index=False)

    # course_list = pd.read_csv(FILE_NAME)

    # adj_matrix = build_adj_matrix(course_list)
    # adj_matrix.to_csv(MATRIX_FILE_NAME)
