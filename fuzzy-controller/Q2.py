# Q2_graded
# Do not change the above line.
!pip install simpful
# Remove this comment and type your codes here

# Q2_graded
# Do not change the above line.

from simpful import *

FS = FuzzySystem()

player_price = AutoTriangle(6, terms=['very cheap', 'cheap', 'normal', 'expensive', 'very_expensive', 'extremely_expensive'], universe_of_discourse=[0,1000])
player_age = AutoTriangle(5, terms=['very_young', 'young', 'middle_age', 'old', 'very_old'], universe_of_discourse=[18,40])
last_5_matches = AutoTriangle(3, terms=['bad', 'medium', 'good'], universe_of_discourse=[0,5])


# Q2_graded
# Do not change the above line.

FS.add_linguistic_variable("price_GoalKeeper_Sepahan", player_price)
FS.add_linguistic_variable("age_GoalKeeper_Sepahan", player_age)
FS.add_linguistic_variable("price_LeftDiffender_Sepahan", player_price)
FS.add_linguistic_variable("age_LeftDiffender_Sepahan", player_age)
FS.add_linguistic_variable("price_RightDiffender_Sepahan", player_price)
FS.add_linguistic_variable("age_RightDiffender_Sepahan", player_age)
FS.add_linguistic_variable("price_MidDiffender1_Sepahan", player_price)
FS.add_linguistic_variable("age_MidDiffender1_Sepahan", player_age)
FS.add_linguistic_variable("price_MidDiffender2_Sepahan", player_price)
FS.add_linguistic_variable("age_MidDiffender2_Sepahan", player_age)
FS.add_linguistic_variable("price_MidFielder1_Sepahan", player_price)
FS.add_linguistic_variable("age_MidFielder1_Sepahan", player_age)
FS.add_linguistic_variable("price_MidFielder2_Sepahan", player_price)
FS.add_linguistic_variable("age_MidFielder2_Sepahan", player_age)
FS.add_linguistic_variable("price_MidFielder3_Sepahan", player_price)
FS.add_linguistic_variable("age_MidFielder3_Sepahan", player_age)
FS.add_linguistic_variable("price_RightForward_Sepahan", player_price)
FS.add_linguistic_variable("age_RightForward_Sepahan", player_age)
FS.add_linguistic_variable("price_LeftForward_Sepahan", player_price)
FS.add_linguistic_variable("age_LeftForward_Sepahan", player_age)
FS.add_linguistic_variable("price_CenterForward_Sepahan", player_price)
FS.add_linguistic_variable("age_CenterForward_Sepahan", player_age)

# Q2_graded
# Do not change the above line.

FS.add_linguistic_variable("price_GoalKeeper_Foolad", player_price)
FS.add_linguistic_variable("age_GoalKeeper_Foolad", player_age)
FS.add_linguistic_variable("price_LeftDiffender_Foolad", player_price)
FS.add_linguistic_variable("age_LeftDiffender_Foolad", player_age)
FS.add_linguistic_variable("price_RightDiffender_Foolad", player_price)
FS.add_linguistic_variable("age_RightDiffender_Foolad", player_age)
FS.add_linguistic_variable("price_MidDiffender1_Foolad", player_price)
FS.add_linguistic_variable("age_MidDiffender1_Foolad", player_age)
FS.add_linguistic_variable("price_MidDiffender2_Foolad", player_price)
FS.add_linguistic_variable("age_MidDiffender2_Foolad", player_age)
FS.add_linguistic_variable("price_MidFielder1_Foolad", player_price)
FS.add_linguistic_variable("age_MidFielder1_Foolad", player_age)
FS.add_linguistic_variable("price_MidFielder2_Foolad", player_price)
FS.add_linguistic_variable("age_MidFielder2_Foolad", player_age)
FS.add_linguistic_variable("price_MidFielder3_Foolad", player_price)
FS.add_linguistic_variable("age_MidFielder3_Foolad", player_age)
FS.add_linguistic_variable("price_RightForward_Foolad", player_price)
FS.add_linguistic_variable("age_RightForward_Foolad", player_age)
FS.add_linguistic_variable("price_LeftForward_Foolad", player_price)
FS.add_linguistic_variable("age_LeftForward_Foolad", player_age)
FS.add_linguistic_variable("price_CenterForward_Foolad", player_price)
FS.add_linguistic_variable("age_CenterForward_Foolad", player_age)

# Q2_graded
# Do not change the above line.

FS.add_linguistic_variable("history_Foolad", last_5_matches)
FS.add_linguistic_variable("history_Sepahan", last_5_matches)

result = AutoTriangle(3, terms=['Sepahan_win', 'tie', 'Foolad_win'], universe_of_discourse=[-10,10])
FS.add_linguistic_variable("result1", result)
FS.add_linguistic_variable("result2", result)
FS.add_linguistic_variable("result3", result)
FS.add_linguistic_variable("result4", result)

# Q2_graded
# Do not change the above line.

Rules = [
        
        "IF (price_GoalKeeper_Sepahan IS expensive) THEN (result1 IS Sepahan_win)",
        "IF (price_GoalKeeper_Foolad IS cheap) THEN (result1 IS Sepahan_win)",  
        "IF (age_GoalKeeper_Sepahan IS middle_age) THEN (result1 IS Sepahan_win)",
        "IF (age_GoalKeeper_Foolad IS very_old) THEN (result1 IS Sepahan_win)",
        "IF (price_GoalKeeper_Foolad IS expensive) THEN (result1 IS Foolad_win)",
        "IF (age_GoalKeeper_Sepahan IS very_old) THEN (result1 IS Foolad_win)",
        
        "IF (price_MidDiffender1_Sepahan IS expensive) THEN (result2 IS Sepahan_win)",
        "IF (price_MidDiffender1_Foolad IS cheap) THEN (result2 IS Sepahan_win)",
        "IF (age_MidDiffender1_Sepahan IS middle_age) THEN (result2 IS Sepahan_win)",
        "IF (age_MidDiffender1_Foolad IS very_old) THEN (result2 IS Sepahan_win)",
        "IF (price_MidDiffender2_Sepahan IS expensive) THEN (result2 IS Sepahan_win)",
        "IF (age_MidDiffender2_Sepahan IS middle_age) THEN (result2 IS Sepahan_win)",
        "IF (age_MidDiffender2_Foolad IS very_old) THEN (result2 IS Sepahan_win)",
        "IF (price_MidDiffender2_Foolad IS cheap) THEN (result2 IS Sepahan_win)",
        "IF (price_LeftDiffender_Sepahan IS expensive) THEN (result2 IS Sepahan_win)",
        "IF (age_LeftDiffender_Sepahan IS middle_age) THEN (result2 IS Sepahan_win)",
        "IF (age_LeftDiffender_Foolad IS very_old) THEN (result2 IS Sepahan_win)",
        "IF (price_LeftDiffender_Foolad IS cheap) THEN (result2 IS Sepahan_win)",
        "IF (price_RightDiffender_Sepahan IS expensive) THEN (result2 IS Sepahan_win)",
        "IF (age_RightDiffender_Sepahan IS middle_age) THEN (result2 IS Sepahan_win)",
        "IF (age_RightDiffender_Foolad IS very_old) THEN (result2 IS Sepahan_win)",
        "IF (price_RightDiffender_Foolad IS cheap) THEN (result2 IS Sepahan_win)",
        "IF (age_MidDiffender1_Foolad IS middle_age) THEN (result2 IS Foolad_win)",
        "IF (age_MidDiffender1_Sepahan IS very_old) THEN (result2 IS Foolad_win)",
        "IF (price_MidDiffender1_Foolad IS expensive) THEN (result2 IS Foolad_win)",
        "IF (price_MidDiffender1_Sepahan IS cheap) THEN (result2 IS Foolad_win)",
        "IF (age_MidDiffender2_Foolad IS middle_age) THEN (result2 IS Foolad_win)",
        "IF (age_MidDiffender2_Sepahan IS very_old) THEN (result2 IS Foolad_win)",
        "IF (price_MidDiffender2_Sepahan IS cheap) THEN (result2 IS Foolad_win)",
        "IF (price_MidDiffender2_Foolad IS expensive) THEN (result2 IS Foolad_win)",
        "IF (price_LeftDiffender_Foolad IS expensive) THEN (result2 IS Foolad_win)",        
        "IF (age_LeftDiffender_Foolad IS middle_age) THEN (result2 IS Foolad_win)",               
        "IF (age_LeftDiffender_Sepahan IS very_old) THEN (result2 IS Foolad_win)",        
        "IF (price_LeftDiffender_Sepahan IS cheap) THEN (result2 IS Foolad_win)",
        "IF (price_RightDiffender_Foolad IS expensive) THEN (result2 IS Foolad_win)",        
        "IF (age_RightDiffender_Foolad IS middle_age) THEN (result2 IS Foolad_win)",       
        "IF (age_RightDiffender_Sepahan IS very_old) THEN (result2 IS Foolad_win)",       
        "IF (price_RightDiffender_Sepahan IS cheap) THEN (result2 IS Foolad_win)",
        
        
        "IF (price_CenterForward_Sepahan IS expensive) THEN (result3 IS Sepahan_win)",
        "IF (age_CenterForward_Sepahan IS middle_age) THEN (result3 IS Sepahan_win)",
        "IF (age_CenterForward_Foolad IS very_old) THEN (result3 IS Sepahan_win)",
        "IF (price_CenterForward_Foolad IS cheap) THEN (result3 IS Sepahan_win)",
        "IF (age_LeftForward_Sepahan IS middle_age) THEN (result3 IS Sepahan_win)",
        "IF (age_LeftForward_Foolad IS very_old) THEN (result3 IS Sepahan_win)",
        "IF (price_LeftForward_Foolad IS cheap) THEN (result3 IS Sepahan_win)",
        "IF (price_LeftForward_Sepahan IS expensive) THEN (result3 IS Sepahan_win)",
        "IF (price_RightForward_Sepahan IS expensive) THEN (result3 IS Sepahan_win)",
        "IF (age_RightForward_Foolad IS very_old) THEN (result3 IS Sepahan_win)",
        "IF (price_RightForward_Foolad IS cheap) THEN (result3 IS Sepahan_win)",
        "IF (price_CenterForward_Foolad IS expensive) THEN (result3 IS Foolad_win)",
        "IF (age_CenterForward_Foolad IS middle_age) THEN (result3 IS Foolad_win)",
        "IF (age_CenterForward_Sepahan IS very_old) THEN (result3 IS Foolad_win)",
        "IF (price_CenterForward_Sepahan IS cheap) THEN (result3 IS Foolad_win)",
        "IF (price_LeftForward_Foolad IS expensive) THEN (result3 IS Foolad_win)",       
        "IF (age_LeftForward_Foolad IS middle_age) THEN (result3 IS Foolad_win)",        
        "IF (age_LeftForward_Sepahan IS very_old) THEN (result3 IS Foolad_win)",        
        "IF (price_LeftForward_Sepahan IS cheap) THEN (result3 IS Foolad_win)",
        "IF (price_RightForward_Foolad IS expensive) THEN (result3 IS Foolad_win)",       
        "IF (age_RightForward_Foolad IS middle_age) THEN (result3 IS Foolad_win)",
        "IF (age_RightForward_Sepahan IS middle_age) THEN (result3 IS Sepahan_win)",       
        "IF (age_RightForward_Sepahan IS very_old) THEN (result3 IS Foolad_win)",      
        "IF (price_RightForward_Sepahan IS cheap) THEN (result3 IS Foolad_win)",
         
        "IF (history_Foolad IS bad) THEN (result4 IS Sepahan_win)",
        "IF (history_Sepahan IS good) THEN (result4 IS Sepahan_win)",
        "IF (history_Foolad IS good) THEN (result4 IS Foolad_win)",
        "IF (history_Sepahan IS bad) THEN (result4 IS Foolad_win)",
        "IF (history_Foolad IS bad) AND (history_Sepahan IS bad) THEN (result4 IS tie)",
        "IF (history_Foolad IS medium) AND (history_Sepahan IS medium) THEN (result4 IS tie)"
]
FS.add_rules(Rules, verbose=True)

# Q2_graded
# Do not change the above line.

FS.set_variable("price_GoalKeeper_Sepahan", 405)
FS.set_variable("age_GoalKeeper_Sepahan", 33)
FS.set_variable("price_LeftDiffender_Sepahan", 1080)
FS.set_variable("age_LeftDiffender_Sepahan", 25)
FS.set_variable("price_RightDiffender_Sepahan", 698)
FS.set_variable("age_RightDiffender_Sepahan", 29)
FS.set_variable("price_MidDiffender1_Sepahan", 450)
FS.set_variable("age_MidDiffender1_Sepahan", 29)
FS.set_variable("price_MidDiffender2_Sepahan", 315)
FS.set_variable("age_MidDiffender2_Sepahan", 31)
FS.set_variable("price_MidFielder1_Sepahan", 585)
FS.set_variable("age_MidFielder1_Sepahan", 20)
FS.set_variable("price_MidFielder2_Sepahan", 495)
FS.set_variable("age_MidFielder2_Sepahan", 29)
FS.set_variable("price_MidFielder3_Sepahan", 450)
FS.set_variable("age_MidFielder3_Sepahan", 32)
FS.set_variable("price_RightForward_Sepahan", 563)
FS.set_variable("age_RightForward_Sepahan", 29)
FS.set_variable("price_LeftForward_Sepahan", 270)
FS.set_variable("age_LeftForward_Sepahan", 33)
FS.set_variable("price_CenterForward_Sepahan", 540)
FS.set_variable("age_CenterForward_Sepahan", 27)

# Q2_graded
# Do not change the above line.

FS.set_variable("history_Sepahan", 4) 
FS.set_variable("history_Foolad", 3) 
FS.set_variable("price_GoalKeeper_Foolad", 270)
FS.set_variable("age_GoalKeeper_Foolad", 27)
FS.set_variable("price_LeftDiffender_Foolad", 338)
FS.set_variable("age_LeftDiffender_Foolad", 23)
FS.set_variable("price_RightDiffender_Foolad", 428)
FS.set_variable("age_RightDiffender_Foolad", 24)
FS.set_variable("price_MidDiffender1_Foolad", 405)
FS.set_variable("age_MidDiffender1_Foolad", 25)
FS.set_variable("price_MidDiffender2_Foolad", 383)
FS.set_variable("age_MidDiffender2_Foolad", 22)
FS.set_variable("price_MidFielder1_Foolad", 225)
FS.set_variable("age_MidFielder1_Foolad", 32)
FS.set_variable("price_MidFielder2_Foolad", 225)
FS.set_variable("age_MidFielder2_Foolad", 29)
FS.set_variable("price_MidFielder3_Foolad", 585)
FS.set_variable("age_MidFielder3_Foolad", 29)
FS.set_variable("price_RightForward_Foolad", 496)
FS.set_variable("age_RightForward_Foolad", 28)
FS.set_variable("price_LeftForward_Foolad", 405)
FS.set_variable("age_LeftForward_Foolad", 22)
FS.set_variable("price_CenterForward_Foolad", 405)
FS.set_variable("age_CenterForward_Foolad", 31)

# Q2_graded
# Do not change the above line.

Sepahan_score = 0
Foolad_score = 0
score = FS.inference()

for i in range(1,5):
    if score[f"result{i}"] > 1:
        Foolad_score += 1
    elif score[f"result{i}"] < -1:
        Sepahan_score += 1

print(f"The prediction of the game is Sepahan {Sepahan_score}-{Foolad_score} Foolad")

