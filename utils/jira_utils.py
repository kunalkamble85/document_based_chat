from jira import JIRA
import re
import json
import traceback
import os

# Jira credentials and server information
JIRA_SERVER = 'https://kunalkamble85.atlassian.net'
JIRA_EMAIL = 'kunalkamble85@gmail.com'
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT_KEY = 'SCRUM'

# Connect to Jira using token authentication
options = {'server': JIRA_SERVER}
jira = JIRA(options, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))


def get_stories_from_llm_response(stories_text):
    try:
        pattern = r"User Story (.*?)\n"
        matches = re.findall(pattern, stories_text, re.MULTILINE)
        story_summaries = [i.split(":")[-1].strip() for i in matches]    
        print(story_summaries)
        pattern = r"\{([^}]*)\}"
        matches = re.findall(pattern, stories_text, re.MULTILINE)
        story_details = []
        counter = 0
        for story_detail in matches:
            details =  story_detail.replace("\n","")       
            details = "{" +details+ "}"
            json_obj = json.loads(details)   
            if len(story_summaries) > counter:         
                json_obj["story_summary"] = story_summaries[counter]
            counter += 1
            story_details.append(json_obj)
        print(story_details)
        return story_details
    except:
        print(traceback.format_exc())
        return [],[]

def embed_jira_into_stories(jira_links, stories_text):
    pattern = r"User Story (.*?)\n"
    matches = re.findall(pattern, stories_text, re.MULTILINE)
    if len(matches) > 0:
        counter = 0
        for match in matches:
            stories_text = stories_text.replace(match, f"{match} ({jira_links[counter]})")
            counter+=1
    else:
        story_links = "\n".join(jira_links)
        stories_text = stories_text + "\n" + story_links
    return stories_text


def create_jira_stories(stories_text):
    try:
        # print(stories_text)
        story_details = get_stories_from_llm_response(stories_text)
        created_stories =[]
        return_text = ""
        counter = 1
        for story in story_details:            
            story_header = story["story_summary"] if "story_summary" in story else story["Summary"]
            at = story["Acceptance_Criteria"]
            at_text = "\n".join(at)
            story_Details = f"""
            Summary:{story["Summary"]}
            Who:{story["Who"]}
            What:{story["What"]}
            Why:{story["Why"]}
            Acceptance_Criteria:\n{at_text}
            """
            issue_dict = {
                'project': {'key': JIRA_PROJECT_KEY},
                'summary': story_header,
                'description': story_Details,
                'issuetype': {'name': 'Story'},
            }
            # print(issue_dict)
            new_issue = jira.create_issue(fields=issue_dict)
            created_stories.append(f"Jira link:{JIRA_SERVER}/browse/{new_issue.key}")           
            at_text ="<ul>"
            for a in at:
               at_text =at_text+"<li>"+a+"</li>"     
            at_text =at_text+"</ul>"
            return_text = f"{return_text}<br>nbsp;nbsp;<strong>User Story {counter}</strong> (Jira link:<a href='{JIRA_SERVER}/browse/{new_issue.key}'>{new_issue.key}</a>)"
            return_text = f"{return_text}<br>nbsp;nbsp;<strong>Summary:</strong> {story_header}"
            return_text = f"{return_text}<br>nbsp;nbsp;<strong>Who:</strong> {story['Who']}"
            return_text = f"{return_text}<br>nbsp;nbsp;<strong>What:</strong> {story['What']}"
            return_text = f"{return_text}<br>nbsp;nbsp;<strong>Why:</strong> {story['Why']}"
            return_text = f"{return_text}<br>nbsp;nbsp;<strong>Acceptance_Criteria:</strong><br>{at_text}"
            return_text = f"{return_text}<br><br>"
            counter+=1
        return return_text
        # stories_text = embed_jira_into_stories(created_stories, stories_text)
    except:
        print(traceback.format_exc())
    return stories_text



a="""
    Here are the detailed level user stories for the given business use case:

    User Story 1: Read CSV File

    {
    "Summary": "As a system, I want to read a CSV file as input so that I can process the data.",
    "Who": "System",
    "What": "Read CSV file",
    "Why": "To process the data from the CSV file",
    "Acceptance_Criteria": [
        "The system can read a CSV file from a specified location.",
        "The system can handle CSV files with varying numbers of columns and rows.",
        "The system can detect and handle errors in the CSV file format."
    ]
    }

    User Story 2: Validate CSV Data

    {
    "Summary": "As a system, I want to validate the data from the CSV file so that I can ensure data quality.",
    "Who": "System",
    "What": "Validate CSV data",
    "Why": "To ensure data quality and prevent errors",
    "Acceptance_Criteria": [
        "The system can validate the data types of each column in the CSV file.",
        "The system can check for missing or empty values in the CSV file.",
        "The system can detect and handle invalid data formats in the CSV file."
    ]
    }

    User Story 3: Transform Data

    {
    "Summary": "As a system, I want to transform the data from the CSV file so that it can be saved in the Oracle database.",
    "Who": "System",
    "What": "Transform data",
    "Why": "To convert the data into a format compatible with the Oracle database",
    "Acceptance_Criteria": [
        "The system can perform data type conversions as required by the Oracle database.",
        "The system can handle data formatting and masking as required by the Oracle database.",
        "The system can detect and handle errors during data transformation."
    ]
    }

    User Story 4: Connect to Oracle Database

    {
    "Summary": "As a system, I want to connect to the Oracle database so that I can save the transformed data.",
    "Who": "System",
    "What": "Connect to Oracle database",
    "Why": "To establish a connection to the Oracle database for data saving",
    "Acceptance_Criteria": [
        "The system can establish a connection to the Oracle database using the provided credentials.",
        "The system can handle connection errors and timeouts.",
        "The system can detect and handle database authentication errors."
    ]
    }

    User Story 5: Save Data to Oracle Database

    {
    "Summary": "As a system, I want to save the transformed data to the Oracle database so that it can be stored and retrieved.",
    "Who": "System",
    "What": "Save data to Oracle database",
    "Why": "To store the transformed data in the Oracle database",
    "Acceptance_Criteria": [
        "The system can save the transformed data to the Oracle database.",
        "The system can handle data insertion errors and rollbacks.",
        "The system can detect and handle database constraints and triggers."
    ]
    }

    These user stories cover the entire process of reading a CSV file, transforming the data, and saving it to an Oracle database. Each user story has a clear summary, who, what, why, and acceptance criteria that define the requirements for the story.
    """
# print(create_jira_stories(stories_text))

