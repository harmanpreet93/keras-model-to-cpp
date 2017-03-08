#include "predict.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

// error for missing implementation of activation function
// you can add your activation implementation in compute_output if required
void missing_activation_impl(const string &activation) {
	cout << "Activation " << activation << " not defined!" << endl;
	cout << "Please add its implementation before use." << endl;
	exit(1);
}

vector<float> read_input_from_file(const string &fname) {
	ifstream fin(fname.c_str());
	int n_features;
	fin >> n_features;
	vector<float> input_data(n_features);
	for (unsigned i = 0; i < n_features; i++) {
		fin >> input_data[i];
	}
	return input_data;
}

int read_response_from_file(const string &fname) {
	ifstream fin(fname.c_str());
	int n_features;
	fin >> n_features;
	float tmp_float;
	for (unsigned i = 0; i < n_features; i++) {
		fin >> tmp_float;
	}
	int response;
	fin >> response;
	return response;
}


// KerasModel constructor
KerasModel::KerasModel(string &input_fname) {
	load_weights(input_fname);
}

// KerasModel destructor
KerasModel::~KerasModel() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];	// deallocate memory
	}
}

// load weights for all layers
void KerasModel::load_weights(string &input_fname) {
	cout << "Reading weights from file " << input_fname << endl;
	ifstream fin(input_fname.c_str(),ifstream::in);
	string tmp_str = "";
	string layer_type = "";
	int layer_id = 0;
	if(fin.is_open()) {
		// get layers count in layers_count var
		fin >> tmp_str >> layers_count;

		// Now iterate over  each layer
		for (unsigned int layer_index = 0; layer_index < layers_count; ++layer_index) {
			fin >> tmp_str >> layer_id >> layer_type;
			// pointer to layer
			Layer *layer = 0L;
			if (layer_type == "Dense") {
				layer = new LayerDense();
			}
			else if(layer_type == "Activation") {
				layer = new LayerActivation();
			}
			// if none of above case is true, means layer not-defined
			if(layer == 0L) {
		      	cout << "Layer is empty, maybe layer " << layer_type << " is not defined? Cannot define network." << endl;
			     return;
			}
			layer->load_weights(fin);
			layers.push_back(layer);
		}
	}
	fin.close();
}

vector<float> KerasModel::compute_output(vector<float> test_input) {
	// cout << "KreasModel compute output" << endl;
	vector<float> response;
	for (unsigned int i = 0; i < layers_count; i++) {
		// cout << "Processing layer to compute output " << layers[i]->layer_name << endl;
		response = layers[i]->compute_output(test_input);
		test_input = response;
	}
	return response;
}

// load weights and bias from input file for Dense layer
void LayerDense::load_weights(ifstream &fin) {
	// cout << "Loading weights for Dense layer" << endl;
	fin >> input_node_count >> output_weights;
	float tmp_float;
	// read weights for all the input nodes
	char tmp_char = ' ';
	for (unsigned int i = 0; i < input_node_count; i++) {
		fin >> tmp_char;	// for '['
		vector<float> tmp_weights;
		for (unsigned int j = 0; j < output_weights; j++) {
			fin >> tmp_float;
			tmp_weights.push_back(tmp_float);
		}
		fin >> tmp_char;	// for ']'
		layer_weights.push_back(tmp_weights);
	}
	// read and save bias values
	fin >> tmp_char;	// for '['
	for (unsigned int output_node_index = 0; output_node_index < output_weights; output_node_index++) {
		fin >> tmp_float;
		bias.push_back(tmp_float);
	}
	fin >> tmp_char;	// for ']'
}

void LayerActivation::load_weights(ifstream &fin) {
	// cout << "Loading weights for Activation layer" << endl;
	fin >> activation_type;
}

vector<float> LayerDense::compute_output(vector<float> test_input) {
	// cout << "Inside dense layer compute output" << '\n';
    // cout << "weights: input size " << layer_weights.size() << endl;
    // cout << "weights: neurons size " << layer_weights[0].size() << endl;
    // cout << "bias size " << bias.size() << endl;
	vector<float> out(output_weights);
	float weighted_term = 0;
	for (size_t i = 0; i < output_weights; i++) {
		weighted_term = 0;
		for (size_t j = 0; j < input_node_count; j++) {
			weighted_term += (test_input[j] * layer_weights[j][i]);
		}
		out[i] = weighted_term + bias[i];
	}
	return out;
}


vector<float> LayerActivation::compute_output(vector<float> test_input) {
	if (activation_type == "linear") {
		return test_input;
	}
	else if(activation_type == "relu") {
		for (unsigned int i = 0; i < test_input.size(); i++) {
			if(test_input[i] < 0) {
				test_input[i] = 0;
			}
		}
	}
	else if(activation_type == "softmax") {
		float sum = 0.0;
        for(unsigned int k = 0; k < test_input.size(); ++k) {
			test_input[k] = exp(test_input[k]);
			sum += test_input[k];
        }

        for(unsigned int k = 0; k < test_input.size(); ++k) {
			test_input[k] /= sum;
        }
	}
	else if (activation_type == "sigmoid") {
		float denominator = 0.0;
		for(unsigned int k = 0; k < test_input.size(); ++k) {
			denominator = 1 + exp(-(test_input[k]));
          	test_input[k] = 1/denominator;
        }
	}
	else if(activation_type == "softplus") {
		for (unsigned int k = 0; k < test_input.size(); ++k) {
			// log1p = natural logarithm (to base e) of 1 plus the given number (ln(1+x))
			test_input[k] = log1p(exp(test_input[k]));
		}
	}
	else if(activation_type == "softsign") {
		for (unsigned int k = 0; k < test_input.size(); ++k) {
			test_input[k] = test_input[k]/(1+abs(test_input[k]));
		}
	}
	else {
      missing_activation_impl(activation_type);
    }
	return test_input;
}
