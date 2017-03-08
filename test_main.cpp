#include "predict.h"

#include <iostream>
using namespace std;

// main function, entry point
int main(int argc, char *argv[]) {

	if(argc != 3) {
		cout << "Wrong input." << endl;
		cout << "The arguments should be: dumped_nn_file input_test_file" << endl;
		return -1;
	}

	string dumped_nn = argv[1];
	string input_test_data = argv[2];

	vector<float> input_data = read_input_from_file(input_test_data);
	int response_class = read_response_from_file(input_test_data);

	cout << "Testing network on " << dumped_nn << ". " << endl;

	// declare keras model object
	KerasModel kerasModel(dumped_nn);
	vector<float> result = kerasModel.compute_output(input_data);

	cout << "Predicted Class: ";
	if(result[0] > 0.5) {
		cout << 1 << endl;
	}
	else {
		cout << 0 << endl;
	}
	cout << "Actual Class: " << response_class << endl;
	cout << "Predicted value: " << result[0] << endl;
	return 0;
}
