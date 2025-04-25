from algo_v9 import detection_fautes
import time
# On récupère les 14 points calibrés au démarrage
with open("court_points.json") as f:
    court_json = f.read()


def Rebond():
    time.sleep(1)
    return True

def get_last_ball_xy():
    return 412.52, 205.46


# Boucle principale (pseudocode) ------------------
while True:
    if Rebond():                         # votre moteur de détection de rebond
        x, y = get_last_ball_xy()        # coordonnées du dernier rebond
        joue = "bottom_player"           # ou "top_player"
        good = detection_fautes(court_json, x, y, joue)
        if good:
            print("Balle IN")
        else:
            print("Faute !")