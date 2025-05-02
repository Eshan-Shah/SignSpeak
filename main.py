from hand_tracking import trackHands
from train_model import trainModel
from collect_data import collectData
from visualise_data import visualiseData
from reset_data import resetData

def main(reset_previous_data=False, collect_new_data=True, visualise_data=False, train_the_model=False):
        
        if reset_previous_data:
              resetData()
        
        if collect_new_data:
            collectData()

        if visualise_data:
            visualiseData()
        
        if train_the_model:
            trainModel()
            
        trackHands()



if __name__ == '__main__':
    main()