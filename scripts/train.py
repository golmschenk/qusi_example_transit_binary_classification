from qusi.model import Hadryss
from qusi.session import TrainHyperparameterConfiguration, train_session

from dataset import get_transit_train_dataset, get_transit_validation_dataset

def main():
    train_light_curve_dataset = get_transit_train_dataset()
    validation_light_curve_dataset = get_transit_validation_dataset()
    model = Hadryss.new()
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=100, cycles=20, train_steps_per_cycle=100, validation_steps_per_cycle=10)
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
                  model=model, hyperparameter_configuration=train_hyperparameter_configuration)


if __name__ == '__main__':
    main()
