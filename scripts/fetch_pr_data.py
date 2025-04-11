#!/usr/bin/env python3
"""
Script to fetch pull request data from GitHub and prepare it for indexing.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime
import json
from typing import Dict, List, Optional
import tempfile
import subprocess

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PR_DATA_DIR = os.path.join(DATA_DIR, "pr_data")

class PRDataFetcher:
    def __init__(self, token: str, repo_owner: str, repo_name: str):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def fetch_pr_list(self, state: str = "all") -> List[Dict]:
        """Fetch list of pull requests."""
        url = f"{self.base_url}/pulls"
        params = {"state": state, "per_page": 100}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def fetch_pr_details(self, pr_number: int) -> Dict:
        """Fetch detailed information about a specific PR."""
        url = f"{self.base_url}/pulls/{pr_number}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def fetch_pr_files(self, pr_number: int) -> List[Dict]:
        """Fetch list of files changed in a PR."""
        url = f"{self.base_url}/pulls/{pr_number}/files"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def fetch_pr_comments(self, pr_number: int) -> List[Dict]:
        """Fetch comments on a PR."""
        url = f"{self.base_url}/issues/{pr_number}/comments"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def fetch_pr_reviews(self, pr_number: int) -> List[Dict]:
        """Fetch reviews on a PR."""
        url = f"{self.base_url}/pulls/{pr_number}/reviews"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_file_diff(self, pr_number: int) -> Dict[str, str]:
        """Get all file diffs for a PR at once."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone the repository
            repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}.git"
            subprocess.run(["git", "clone", repo_url, temp_dir], check=True)
            
            # Fetch the PR branch
            os.chdir(temp_dir)
            subprocess.run(["git", "fetch", "origin", f"pull/{pr_number}/head:pr-{pr_number}"], check=True)
            subprocess.run(["git", "checkout", f"pr-{pr_number}"], check=True)
            
            # Get all changed files
            result = subprocess.run(
                ["git", "diff", "--name-only", "origin/main"],
                capture_output=True,
                text=True
            )
            changed_files = result.stdout.strip().split('\n')
            
            # Get diff for each file
            file_diffs = {}
            for file_path in changed_files:
                if file_path:  # Skip empty lines
                    result = subprocess.run(
                        ["git", "diff", "origin/main", "--", file_path],
                        capture_output=True,
                        text=True
                    )
                    file_diffs[file_path] = result.stdout
        
            return file_diffs

def process_pr_data(pr_data: Dict, fetcher: PRDataFetcher) -> Dict:
    """Process PR data into a format suitable for indexing."""
    pr_number = pr_data["number"]
    
    # Fetch additional PR details
    files = fetcher.fetch_pr_files(pr_number)
    comments = fetcher.fetch_pr_comments(pr_number)
    reviews = fetcher.fetch_pr_reviews(pr_number)
    
    # Get all file diffs at once
    print("  Fetching all file diffs...")
    file_diffs = fetcher.get_file_diff(pr_number)
    print(f"  Found {len(file_diffs)} changed files")
    
    # Process files with diffs
    processed_files = []
    for file in files:
        try:
            diff = file_diffs.get(file["filename"], "")
            
            # Split diff into chunks for better LLM processing
            diff_chunks = []
            current_chunk = []
            for line in diff.split('\n'):
                if line.startswith('@@') and current_chunk:
                    # Start new chunk when we hit a new hunk
                    diff_chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                current_chunk.append(line)
            if current_chunk:
                diff_chunks.append('\n'.join(current_chunk))
            
            # Create a summary of changes
            summary = {
                "total_changes": file["additions"] + file["deletions"],
                "additions": file["additions"],
                "deletions": file["deletions"],
                "file_type": os.path.splitext(file["filename"])[1],
                "status": file["status"]
            }
            
            processed_files.append({
                "filename": file["filename"],
                "summary": summary,
                "diff_chunks": diff_chunks,
                "full_diff": diff  # Keep full diff for reference
            })
        except Exception as e:
            print(f"Error processing file {file['filename']}: {e}")
    
    # Create a summary of the PR
    pr_summary = {
        "total_files_changed": len(processed_files),
        "total_additions": sum(f["summary"]["additions"] for f in processed_files),
        "total_deletions": sum(f["summary"]["deletions"] for f in processed_files),
        "file_types": list(set(f["summary"]["file_type"] for f in processed_files)),
        "main_changes": [f["filename"] for f in processed_files if f["summary"]["total_changes"] > 10]
    }
    
    return {
        "pr_number": pr_number,
        "title": pr_data["title"],
        "description": pr_data["body"],
        "state": pr_data["state"],
        "created_at": pr_data["created_at"],
        "updated_at": pr_data["updated_at"],
        "author": pr_data["user"]["login"],
        "summary": pr_summary,
        "files": processed_files,
        "comments": [{
            "user": comment["user"]["login"],
            "body": comment["body"],
            "created_at": comment["created_at"]
        } for comment in comments],
        "reviews": [{
            "user": review["user"]["login"],
            "state": review["state"],
            "body": review["body"],
            "submitted_at": review["submitted_at"]
        } for review in reviews]
    }

def main():
    """Main function to fetch and process PR data."""
    print("Starting PR data fetch...")
    
    # Check for required environment variables
    required_vars = ["GITHUB_TOKEN", "GITHUB_REPO_OWNER", "GITHUB_REPO_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or export them.")
        sys.exit(1)
    
    print("Environment variables check passed")
    print(f"Repository: {os.getenv('GITHUB_REPO_OWNER')}/{os.getenv('GITHUB_REPO_NAME')}")
    
    # Create PR data directory
    os.makedirs(PR_DATA_DIR, exist_ok=True)
    print(f"PR data directory: {PR_DATA_DIR}")
    
    # Initialize PR fetcher
    print("Initializing PR fetcher...")
    fetcher = PRDataFetcher(
        token=os.getenv("GITHUB_TOKEN"),
        repo_owner=os.getenv("GITHUB_REPO_OWNER"),
        repo_name=os.getenv("GITHUB_REPO_NAME")
    )
    
    try:
        # Specific PR numbers to fetch
        target_prs = [1440, 1441]
        print(f"\nFetching specific pull requests: {target_prs}")
        
        # Process each target PR
        for pr_number in target_prs:
            print(f"\nProcessing PR #{pr_number}")
            
            try:
                print("  Fetching PR details...")
                # Fetch PR details directly
                pr_data = fetcher.fetch_pr_details(pr_number)
                print(f"  Found PR: {pr_data['title']}")
                
                # Process PR data
                processed_data = process_pr_data(pr_data, fetcher)
                
                # Save to file
                output_file = os.path.join(PR_DATA_DIR, f"pr_{pr_number}.json")
                print(f"  Saving PR data to {output_file}")
                with open(output_file, "w") as f:
                    json.dump(processed_data, f, indent=2)
                
                print(f"  Successfully saved PR #{pr_number} data")
            except Exception as e:
                print(f"Error processing PR #{pr_number}: {e}")
                print(f"Error details: {str(e)}")
        
        print("\nPR data fetching completed successfully!")
        print(f"Data saved in: {PR_DATA_DIR}")
    except Exception as e:
        print(f"Error during PR data fetching: {e}")
        print(f"Error details: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 