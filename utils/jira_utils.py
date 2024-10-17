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
        pattern = r"```json(.*?)```"
        match = re.search(pattern, stories_text, re.DOTALL)
        if match:
            tasks_string = match.group(1).strip()
        else:
            pattern = r"```(.*?)```"
            match = re.search(pattern, stories_text, re.DOTALL)
            if match:
                tasks_string = match.group(1).strip()
            else:
                tasks_string = stories_text
        json_obj = json.loads(tasks_string)   
        return json_obj    
    except:
        print(traceback.format_exc())
        return None


def create_jira_subtasks(tasks, parent):
    task_tickets = {}
    try:
        for task in tasks:
            role = task["role"]
            if role == "Business Analyst": role = "BA"
            if role == "Quality Assurance": role = "QA"
            summary = task["task"]
            task_str = f"{role}: {summary}"
            issue_dict = {
                'project': {'key': JIRA_PROJECT_KEY},
                'summary': task_str,
                'description': task_str,
                'issuetype': {'name': 'Subtask'},
                "parent":{"key": parent}
            }
            new_issue = jira.create_issue(fields=issue_dict)
            task_tickets[task_str] = new_issue.key
    except:
        print(traceback.format_exc())
        print("Error while creating sub tasks.")
    return task_tickets


def create_jira_stories(stories_text):
    try:
        # print(stories_text)
        story_details = get_stories_from_llm_response(stories_text)
        created_stories =[]
        return_text = ""
        counter = 1
        for story in story_details:            
            story_header = story["Summary"]
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
            jira_tickets = create_jira_subtasks(story["Tasks"], new_issue.key) 
            at_text =""
            for a in at:
               at_text = at_text + "<br>      -> "+ a
            return_text = f"{return_text}<br><strong> User Story {counter}</strong> (Jira link:<a href='{JIRA_SERVER}/browse/{new_issue.key}'>{new_issue.key}</a>)"
            return_text = f"{return_text}<br><strong> Summary:</strong> {story_header}"
            return_text = f"{return_text}<br><strong> Who:</strong> {story['Who']}"
            return_text = f"{return_text}<br><strong> What:</strong> {story['What']}"
            return_text = f"{return_text}<br><strong> Why:</strong> {story['Why']}"
            return_text = f"{return_text}<br><strong> Acceptance_Criteria:</strong>{at_text}"
            return_text = f"{return_text}<br><strong> Sub-Tasks:</strong>"
            for task, link in jira_tickets.items():
                return_text = f"{return_text}<br>&nbsp;&nbsp;&nbsp;&nbsp;<strong>{task}</strong> (Jira link:<a href='{JIRA_SERVER}/browse/{link}'>{link}</a>)"

            return_text = f"{return_text}<br>---------------------------------------------------------------------------------------------------------------------<br>"
            counter+=1
        return return_text
    except:
        print(traceback.format_exc())
    return stories_text

a = """"""
# print(get_stories_from_llm_response(a))

