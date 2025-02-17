import json

"""
Terrain de tennis :
l'objectif est de détecter les fautes au tennis, donc la position de la balle est importante.
On a un terrain de tennis avec des coordonnées de points, et on veut savoir si la balle est sortie du terrain.
Notre terrain est filmé avec un angle de vue qui déforme les coordonnées des points.
On a donc besoin de déformer les coordonnées de la balle pour les comparer avec celles du terrain.

On consière ces coordonnées de terrain pour nos tests, les calculs sont fait en fonction des coordonnées unstretchées. :
Before unstretch_court:
X_0 = 387, Y_0 = 146
X_1 = 877, Y_1 = 150
X_2 = 89, Y_2 = 509
X_3 = 1189, Y_3 = 508
X_4 = 448, Y_4 = 149
X_5 = 225, Y_5 = 507
X_6 = 819, Y_6 = 151
X_7 = 1045, Y_7 = 507
X_8 = 424, Y_8 = 190
X_9 = 841, Y_9 = 189
X_10 = 318, Y_10 = 364
X_11 = 952, Y_11 = 365
X_12 = 632, Y_12 = 195
X_13 = 634, Y_13 = 363

After unstretch_court:
X_0 = 89, Y_0 = 146
X_1 = 1189, Y_1 = 150
X_2 = 89, Y_2 = 509
X_3 = 1189, Y_3 = 508
X_4 = 225, Y_4 = 149
X_5 = 225, Y_5 = 507
X_6 = 1045, Y_6 = 151
X_7 = 1045, Y_7 = 507
X_8 = 225, Y_8 = 190
X_9 = 1045, Y_9 = 189
X_10 = 225, Y_10 = 364
X_11 = 1045, Y_11 = 365
X_12 = 634, Y_12 = 195
X_13 = 634, Y_13 = 363

           4                              6
      0 +==+==============================+==+ 1
        |  |                              |  |
        |  |                              |  |
        |  |                              |  |
        |  |                              |  |
        |  |              12              |  |
        | 8|==============================|9 |
        |  |              |               |  |
        |  |              |               |  |
        |  |              |               |  |
        |  |              |               |  |
        |  |              |               |  |
     o==+==+==============|===============+==+==o
        |  |              |               |  |  
        |  |              |               |  |
        |  |              |               |  |
        |  |              |               |  |
        |  |              |               |  |
        |10|==============================|11|
        |  |              13              |  |
        |  |                              |  |
        |  |                              |  |
        |  |                              |  |
        |  |                              |  |
      2 +==+==============================+==+ 3
           5                              7
"""

court_data = '''
{
    "points": [
        {"x": 387, "y": 146},
        {"x": 877, "y": 150},
        {"x": 89, "y": 509},
        {"x": 1189, "y": 508},
        {"x": 448, "y": 149},
        {"x": 225, "y": 507},
        {"x": 819, "y": 151},
        {"x": 1045, "y": 507},
        {"x": 424, "y": 190},
        {"x": 841, "y": 189},
        {"x": 318, "y": 364},
        {"x": 952, "y": 365},
        {"x": 632, "y": 195},
        {"x": 634, "y": 363}
    ]
}
'''

data = json.loads(court_data)
points = data['points']

# Assignation des valeurs aux variables
X_0, Y_0 = points[0]['x'], points[0]['y']
X_1, Y_1 = points[1]['x'], points[1]['y']
X_2, Y_2 = points[2]['x'], points[2]['y']
X_3, Y_3 = points[3]['x'], points[3]['y']
X_4, Y_4 = points[4]['x'], points[4]['y']
X_5, Y_5 = points[5]['x'], points[5]['y']
X_6, Y_6 = points[6]['x'], points[6]['y']
X_7, Y_7 = points[7]['x'], points[7]['y']
X_8, Y_8 = points[8]['x'], points[8]['y']
X_9, Y_9 = points[9]['x'], points[9]['y']
X_10, Y_10 = points[10]['x'], points[10]['y']
X_11, Y_11 = points[11]['x'], points[11]['y']
X_12, Y_12 = points[12]['x'], points[12]['y']
X_13, Y_13 = points[13]['x'], points[13]['y']


def unstretch_court():
    global X_0, X_1, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13
    X_0 = X_2
    X_10 = X_5
    X_8 = X_5
    X_4 = X_5
    X_12 = X_13
    X_11 = X_7
    X_9 = X_7
    X_6 = X_7
    X_1 = X_3

def is_bottom_player(player):
    if (player == "bottom_player"):
        return True
    else:
        return False

def is_inside_solo_north_court(ball_X, ball_Y):
    return X_4 <= ball_X <= X_6 and Y_4 <= ball_Y <= (Y_5 / 2 + Y_4 / 2)

def is_inside_double_north_court(ball_X, ball_Y):
    return X_0 <= ball_X <= X_1 and Y_4 <= ball_Y <= (Y_5 / 2 + Y_4 / 2)

def is_inside_solo_south_court(ball_X, ball_Y):
    return X_4 <= ball_X <= X_6 and (Y_5 / 2 + Y_4 / 2) <= ball_Y <= Y_5 

def is_inside_double_south_court(ball_X, ball_Y):
    return X_0 <= ball_X <= X_1 and (Y_5 / 2 + Y_4 / 2) <= ball_Y <= Y_5

""" Les fonctions is_inside_solo_north... sont bonnes mais peut etre mal utilisée dans is_ball_inside_solo_court..., j ai inversé north et south dans is_ball_inside_solo_court..."""

def is_ball_inside_solo_court(ball_X, ball_Y, player):
    if is_bottom_player(player):
        return is_inside_solo_north_court(ball_X, ball_Y)
    else:
        return is_inside_solo_south_court(ball_X, ball_Y)

def is_ball_inside_double_court(ball_X, ball_Y, player):
    if is_bottom_player(player):
        return is_inside_double_north_court(ball_X, ball_Y)
    else:
        return is_inside_double_south_court(ball_X, ball_Y)

def test():
    bottom_player = "bottom_player"
    top_player = "top_player"

    print("Before unstretch_court:")
    for i in range(14):
        print(f'X_{i} = {globals()[f"X_{i}"]}, Y_{i} = {globals()[f"Y_{i}"]}')

    unstretch_court()

    print("\nAfter unstretch_court:")
    for i in range(14):
        print(f'X_{i} = {globals()[f"X_{i}"]}, Y_{i} = {globals()[f"Y_{i}"]}')

    print("\nTest balles in ou out")
    
    # Balles pour les tests
    # Bonne balle pour le joueur du bas
    ball1_X, ball1_Y = 500, 200
    # Mauvaise balle pour le joueur du bas
    ball2_X, ball2_Y = 200, 100
    # Bonne balle pour le joueur du haut
    ball3_X, ball3_Y = 600, 500
    # Mauvaise balle pour le joueur du haut
    ball4_X, ball4_Y = 1200, 400

    print("\nTest des fonctions:")
    print(f'Ball 1 inside solo court (bottom player): {is_ball_inside_solo_court(ball1_X, ball1_Y, bottom_player)}')
    print(f'Ball 2 outside solo court (bottom player): {not is_ball_inside_solo_court(ball2_X, ball2_Y, bottom_player)}')
    print(f'Ball 3 inside solo court (top player): {is_ball_inside_solo_court(ball3_X, ball3_Y, top_player)}')
    print(f'Ball 4 outside solo court (top player): {not is_ball_inside_solo_court(ball4_X, ball4_Y, top_player)}')

    print("\nVoici (Y_5 / 2 + Y_4 / 2) pour voir si j ai bien le nieau du filet")
    print("Donc deja on a Y_4 qui vaut 149 : ", Y_4)
    print("Et Y_5 qui vaut 507 : ", Y_5)
    print("Donc je vais faire Y_5 - Y_4 pour mettre a 0 les coordonnées, Y_5 vaut donc 358 : ", Y_5 - Y_4)
    print("Et je divise par 2 pour avoir le milieu du filet, donc Y_5 / 2 vaut 179 : ", ((Y_5 - Y_4) / 2))
    print("Voici la position de la balle X")
    print("Il faut que je fasse pareil avec la balle pour voir si elle est au dessus ou en dessous du filet precisement, donc on change le Y de la balle pour enlever Y_4 ca fait 200 - 149 = 51 :", ball1_Y - Y_4)
    print((Y_5 / 2 + Y_4 / 2))

    print("\nVoici (Y_5 / 2 + Y_4 / 2) : ", Y_5 / 2 + Y_4 / 2)

    # Affichage des coordonnées des balles pour vérification
    print("\nCoordonnées des balles:")
    print(f'Ball 1: X = {ball1_X}, Y = {ball1_Y}')
    print(f'Ball 2: X = {ball2_X}, Y = {ball2_Y}')
    print(f'Ball 3: X = {ball3_X}, Y = {ball3_Y}')
    print(f'Ball 4: X = {ball4_X}, Y = {ball4_Y}')

    # Affichage des limites du terrain pour vérification
    print("\nLimites du terrain après unstretch_court:")
    print(f'X_4 = {X_4}, X_6 = {X_6}, Y_4 = {Y_4}, Y_5 = {Y_5}')
    
test()

"""
def is_ball_outside_solo_north_court():
    if (X_balle < X_4_8 ou X_balle > X_6_9):
        return 1
    else if (Y_balle < Y_4_6 ou Y_balle > (Y_5_7 - Y_4_6)/2 + Y_4_6):
        return 1
    else:
        return 0

def is_ball_outside_solo_south_court():
    if (X_balle < X_4_8 ou X_balle > X_6_9):
        return 1
    else if (Y_balle > Y_5_7 ou Y_balle <  (Y_5_7 - Y_4_6)/2 + Y_4_6):
        return 1
    else:
        return 0

def Analyse_Rebond():
    truc_complexe.run()
    if(ca_rebond):
        return 1
    else:  
        return 0

def is_playerB_turn():
    if(Analyse_Rebond()):
        if(!is_ball_outside_solo_north_court()):
            return 1
        else
            return 0

def is_playerA_turn():
    if(Analyse_Rebond()):
        if(!is_ball_outside_solo_south_court()):
            return 1
        else
            return 0
"""