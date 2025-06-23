import sys
import torch

def average_weights(file_list, output_file):
    if not file_list:
        print("No files provided for averaging.")
        return

    # Load all state dictionaries
    state_dicts = [torch.load(file, map_location=torch.device('cpu')) for file in file_list]

    # Initialize the averaged state dictionary using the keys of the first model
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        # Sum the corresponding weights from all models and then average
        avg_state_dict[key] = sum(state_dict[key] for state_dict in state_dicts) / len(state_dicts)

    # Save the averaged weights to the output file
    torch.save(avg_state_dict, output_file)
    print(f"Averaged weights saved to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python weight_averaging.py <output_file.pth> <model1.pth> <model2.pth> ...")
        sys.exit(1)

    output_file = sys.argv[1]
    file_list = sys.argv[2:]
    average_weights(file_list, output_file)
