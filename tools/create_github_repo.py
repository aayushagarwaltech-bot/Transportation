import os
import sys
import requests

def main():
    if len(sys.argv) < 4:
        print("Usage: create_github_repo.py <token> <repo_name> <visibility>")
        sys.exit(2)
    token = sys.argv[1]
    repo = sys.argv[2]
    visibility = sys.argv[3]
    private = visibility.lower() != 'public'

    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    data = {"name": repo, "private": private}
    r = requests.post('https://api.github.com/user/repos', json=data, headers=headers)
    if r.status_code not in (201,):
        print(f"Failed to create repo: {r.status_code} {r.text}")
        sys.exit(1)
    resp = r.json()
    owner = resp['owner']['login']
    clone_url = resp['clone_url']
    print(owner)
    # print clone_url to stdout as well (without token)
    print(clone_url)

if __name__ == '__main__':
    main()
