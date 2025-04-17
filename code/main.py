from JuniaProjetTennis.code.Ball.position_ball import ball

# main function
if __name__ == '__main__':
    ball(
            output_dir="../../JuniaProjetTennis/assets/",
            video_path="../Vidéos/tennis.mp4",
            model_path="../Ball/best2.pt"
    )