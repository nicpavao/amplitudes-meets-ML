# amplitudes-meets-ML
Shared repo for some physicists to code up some Boltzmann Brains. I'm thinking we just start with some simple projects:

1) ~~Use PyTorch to fit BRK.A ticker to an exponential curve?~~
2) ~~Use a neural nets (feed forward, CNN, Transformers) to classify digits numerically.~~
3) Use transformer networks to predict tokens (letters/words/integers) in sequential data.

But first, some best practices for version control so we can speak the langauge of industry data science and ML. 

## Step 1: Setting up git
In order to use the full functionality of GitHub, we first need to setup git... To do this we will install homebrew, which is a package manager for MacOS. Copy and paste the following into your terminal:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Now that brew is installed, use the brew command to install git:
```
brew install git
```
Verify that version is at least 2.28 by running
```
git --version
```
I'll assume that your package manager installed at least 2.28 version of git. If not, let's chat.

## Step 2: Configuring git with GitHub, and setting up SSH key
When hosting a shared repository on GitHub, it is useful to setup a SSH cryptographic key, so you don't have to re-enter your password all the time. Let's configure out version control software, git, with the git enabled online platform GitHub. First, enter your username and email for GitHub:
```
git config --global user.name "Your Name"
git config --global user.email "yourname@example.com"
```
Now change the default branch to main, and change the reconciliation behavior to merging:
```
git config --global init.defaultBranch main
git config --global pull.rebase false
```
Finally, for Mac users, run the following commands:
```
echo .DS_Store >> ~/.gitignore_global
git config --global core.excludesfile ~/.gitignore_global
```
We're almost done. To avoid entering your username and password everytime you want to pull from the global main branch, we are going to set up an SSH key. Instead of writing it all out here, just checkout TheOdinProject (TOP) guided instructions here: https://www.theodinproject.com/lessons/foundations-setting-up-git#step-23-create-an-ssh-key

Now we're ready to start coding!

## Step 3: Git Clone, Push, Pull, etc.
Best practices for version control when working in a group. Forking a branch, pull requests, etc.

Follow these steps to create a branch and make changes. This will result in minimal merge conflicts. Failure to follow steps will lead to unneccessary pain.
1. Checkout your feature branch from main
```
git checkout main
git pull
git checkout -b username/feature #ex: nic/dev-branch
```
2. Make your changes on your branch
3. Commit your changes
```
git add .
git commit -m "your message here"
```
4. Make as many commits as you like
5. When ready to put up a pull request or sync up with main
```
git checkout main
git pull origin main
git checkout your-branch-name
git rebase -i main
```
6. At this point you will be shown a list of all your commits. You can squash/fixup/rename as you like. Once your PR gets merged it will automatically squash all commits into one. I like to rename the first commit by changing "pick" to "r" and merge all additional commits into that one by changing "pick" to "f". This will merge all commits into one for easy merging.
Before:
```
pick e682d5d First commit
pick 802af36 second
pick 967ca3e third
```
After
```
r e682d5d First commit
f 802af36 second
f 967ca3e third
```
7. Resolve all merge conflicts. I recommend using VSCode to resolve all the diffs.
8. Push your changes (on the dev branch) to github (which has been locally rebased onto the main branch)
```
git push origin your-branch-name
```
9. Open github and make a pull request

In some cases, one of us might decide to make a comment on a pull request before merging. Maybe there's a bug,
or something was overwritten. If that happens, then...

10. If addressing comments add a new commit to your branch
```
git add .
git commit -m "Address review comments"
```
11. Push your branch to github using force push (only use this on your feature branch and make sure your local changes are correct before overwriting the branch in github)
```
git push origin your-branch-name -f
```
12. Repeat until PR review finished
13. Merge PR using squash commit so all commits are merged into one.

## Step 4: Setting up virtual environments
Sometimes project collaborators will have different versions of the project depencies downloaded on their local machine. For this reason, it is useful to set up a virtual environment where dependencies (like PyTorch, MatplotLib, etc.) can be accessed locally, and shared across machines.

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
requirements.txt file is just a line by line list of libraries

## Definitions
