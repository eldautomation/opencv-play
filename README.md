# opencv-play
This is the readme file

# testing
pytest -m integration --keep-output# 
# runs from project root
# runs all tests marked with integration
# keeps the output - output goes here: /outputs/integration/center_finding


pytest -m integration --keep-output# 
# runs from project root
# runs all unit tests

# tests are located in ./project_root/tests
# see eparate foldr for unit & integration. 
# source files are in ./project_root/tests/assets - note there are different kinds of assets. 

/src/autocollimator/target_utils - source file for target finding info.


# SSH agent usage. 
$ eval "$(ssh-agent -s)" # This starts the ssh agent. 

ssh-add <key> # this adds the pass key, so you can do stuff. 


