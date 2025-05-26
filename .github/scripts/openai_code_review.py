import os

import openai
from github import Github

# Setup
openai.api_key = os.environ["OPENAI_API_KEY"]
g = Github(os.environ["GITHUB_TOKEN"])
repo = g.get_repo(os.environ["GITHUB_REPO"])
pr = repo.get_pull(int(os.environ["PR_NUMBER"]))
prompt_base = os.environ.get(
    "CODE_REVIEW_PROMPT", "Review this code for bugs, performance, and clarity."
)
system_msg = os.environ.get(
    "CODE_REVIEW_SYSTEM_MESSAGE", "You are a helpful senior developer reviewing code."
)

# Collect PR file diffs
comments = []
for f in pr.get_files():
    if f.filename.endswith((".py", ".cs")) and f.patch:
        diff = f.patch[:3000]  # truncate to avoid token limit
        prompt = f"{prompt_base}\n\nFile: {f.filename}\n\n```diff\n{diff}\n```"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        review = response["choices"][0]["message"]["content"]
        comments.append(f"### Review for `{f.filename}`\n{review}\n")
    elif f.filename.endswith((".md", ".rst")) and f.patch:
        diff = f.patch[:3000]
        prompt = f"{prompt_base}\n\nFile: {f.filename}\n\n```diff\n{diff}\n```"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": "fix typos and improve clarity in this code:\n",
                },
            ],
            max_tokens=300,
        )
        review = response["choices"][0]["message"]["content"]
        comments.append(f"### Review for `{f.filename}`\n{review}\n")

# Post comment to PR
if comments:
    pr.create_issue_comment("## 🤖 AI Code Review\n" + "\n---\n".join(comments))
