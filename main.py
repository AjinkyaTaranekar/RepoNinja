from repo_ninja import RepoNinja

github_url = input(
    "🐙 Enter GitHub URL:\nExample (https://github.com/AjinkyaTaranekar/Imagico): "
)
branch = input("🌿 Enter branch name: ")
directories = input("📁 Enter comma separated directories name: ")


user, repo = github_url.split("/")[-2:]
print("⏳ Starting initiation of Ninja ⏳")
repo_ninja = RepoNinja(user, repo, branch, directories)

# [
#     "What is the main purpose of this repository?",
#     "What is the flow architecture of this repository?",
#     "Explain design pattern used in this repository?",
#     "Explain any loophole you can found in this?",
#     "What all features are there in this repository?",
# ]

print(
    f"🥷 Repo Ninja initiated for repo:{user}/{repo}, \nbranch:{branch}, \ndirectories:{directories}"
)
while True:
    print(
        "-------------------------------------------------------------------------------"
    )
    query = input("🤔 Enter your query: ")
    answer = repo_ninja.answer_query(query)
    print(
        f"\n\n📝 Answer to query: \n{answer['responses']} \n\nℹ️  Sources: {answer['sources']}"
    )
    print(
        "-------------------------------------------------------------------------------"
    )

    should_continue = input("Do you want to continue? (y/n): ")
    if should_continue == "n":
        break
