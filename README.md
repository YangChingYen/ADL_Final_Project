# ADL_Final_Project


### Git Usage

1. Fork on GitHub page

2. Clone your GitHub repo

3. Add upstream : 
   ```
   git remote add upstream git@github.com:YangChingYen/ADL_Final_Project.git
   ```

* If you want to pull others' code :  
  ```
  git pull upstream main
  ```

* Push to upstream : 

  ```
  git push origin main
  ```
  Then go to GitHub page to send a pull request

### Get Pairs of Similar prompts.
```
pip install tensorflow tensorflow_hub
cd cos_similarity_test
python classify.py
```
the result is stored at **similar_pair.json**
