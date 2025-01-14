import re
import json

def extract_answers(text):
    """
    Extracts answers from text formatted with Person, Questions, and Answers.

    Args:
        text (str): The input text containing Persons, Questions, and Answers.

    Returns:
        dict: A nested dictionary with persons as keys and their Q&A as values.
    """
    # Regex pattern to capture each person's section
    person_sections = re.split(r"Person:\s*(\w+)", text)
    extracted_data = {}

    for i in range(1, len(person_sections) - 1, 2):
        person = person_sections[i].strip()
        content = person_sections[i + 1].strip()
        
        # Extract each question and corresponding answer
        qa_pairs = re.findall(
            r"Question\s*\d+:\s*(.*?)\nAnswer:\s*(.*?)(?=\nQuestion\s*\d+:|$)", content, re.DOTALL
        )

        # Organize the extracted Q&A pairs into a dictionary
        extracted_data[person] = {f"Question {idx + 1}": {"Question": q, "Answer": a.strip()} for idx, (q, a) in enumerate(qa_pairs)}

    return extracted_data

# Example usage
text = """
Person: Jake
Question 1: Describe the mission of your group and its major accomplishments.
Answer: Our group’s mission is to support seniors in our neighborhood by providing them with free cleaning supplies. Since our formation in 2020, we have distributed hundreds of cleaning supply kits to seniors on fixed incomes who often struggle to afford these essential items. By offering this service, we help alleviate financial burdens and contribute to a healthier living environment for older adults. One community member shared, "I am so grateful for this group and their efforts in providing us with the necessary cleaning supplies. It's one less thing for us to worry about during these difficult times."

Question 2: What is your project and how will it extend beyond your regular operations?
Answer: To support our mission, we have planned a project that involves hosting a series of 4 events at senior centers and community resource centers located in and around the Manhattan Valley section of the Upper West Side. These events will involve 10 to 20 volunteers at each site who will assist us in assembling individual cleaning supply kits for seniors in need. Each event will aim to prepare 50-100 kits, depending on the availability of supplies and volunteers. We will even put notes and letters in each kit with an encouraging message in the hopes that the people who receive the kits will know that they are seen and appreciated.

Person: Sarah
Question 1: Describe the mission of your group and its major accomplishments.
Answer: Our group is dedicated to improving literacy among underserved children in our community. We have held after-school tutoring sessions for over 200 children since our inception. Through this initiative, we have seen measurable improvement in reading comprehension scores and overall academic confidence. One child we worked with, Jake, improved his reading grade from a C to an A in just a few months.

Question 2: What is your project and how will it extend beyond your regular operations?
Answer: Our project involves expanding our tutoring services to reach a wider age range, from elementary to middle school students. We will recruit more volunteers, utilize digital resources, and provide free reading materials to the children who attend. This project will extend beyond regular tutoring by hosting workshops and creating a mobile app for easy access to learning resources for parents and students.
"""

# Extract answers from the text
answers_data = extract_answers(text)

# Save to JSON file
output_file = "extracted_answers.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(answers_data, f, indent=4)

print(f"Extracted answers saved to {output_file}")
