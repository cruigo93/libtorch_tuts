#include <ATen/ATen.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <string>
#include <dirent.h>

using namespace std;
torch::Tensor read_data(std::string loc){
	cv::Mat img = cv::imread(loc, 0);
	cv::resize(img, img, cv::Size(200, 200), cv::INTER_CUBIC);
	std::cout << "Sizes: " << img.size() << std::endl;
	torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 1}, torch::kByte);
	img_tensor = img_tensor.permute({2, 0, 1});
	return img_tensor.clone();
}

torch::Tensor read_label(int label){
	torch::Tensor label_tensor = torch::full({1}, label);
	return label_tensor.clone();
}

vector<torch::Tensor> process_images(vector<string> list_images) {
	cout << "Reading images..." << endl;
	vector<torch::Tensor> states;
	for (std::vector<string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
        cout << "Location being read: " << *it << endl;
		torch::Tensor img = read_data(*it);
		states.push_back(img);
	}
	cout << "Reading and Processing images done!" << endl;
	return states;
}

vector<torch::Tensor> process_labels(vector<int> list_labels) {
	cout << "Reading labels..." << endl;
	vector<torch::Tensor> labels;
	for (std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
		torch::Tensor label = read_label(*it);
		labels.push_back(label);
	}
	cout << "Labels reading done!" << endl;
	return labels;
}

/* This function returns a pair of vector of images paths (strings) and labels (integers) */
std::pair<vector<string>,vector<int>> load_data_from_folder(vector<string> folders_name) {
	vector<string> list_images;
	vector<int> list_labels;
	int label = 0;
	for(auto const& value: folders_name) {
		string base_name = value + "/";
		cout << "Reading from: " << base_name << endl;
		DIR* dir;
		struct dirent *ent;
		if((dir = opendir(base_name.c_str())) != NULL) {
			while((ent = readdir(dir)) != NULL) {
				string filename = ent->d_name;
				if(filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
					cout << base_name + ent->d_name << endl;
					// cv::Mat temp = cv::imread(base_name + "/" + ent->d_name, 1);
					list_images.push_back(base_name + ent->d_name);
					list_labels.push_back(label);
				}

			}
			closedir(dir);
		} else {
			cout << "Could not open directory" << endl;
			// return EXIT_FAILURE;
		}
		label += 1;
	}
	return std::make_pair(list_images, list_labels);
}

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
	/* data */
	// Should be 2 tensors
	vector<torch::Tensor> states, labels;
public:
	CustomDataset(vector<string> list_images, vector<int> list_labels) {
		states = process_images(list_images);
		labels = process_labels(list_labels);
	};

	torch::data::Example<> get(size_t index) override {
		/* This should return {torch::Tensor, torch::Tensor} */
		torch::Tensor sample_img = states.at(index);
		torch::Tensor sample_label = labels.at(index);
		return {sample_img.clone(), sample_label.clone()};
	};

  torch::optional<size_t> size() const override {
		return states.size();
  };
};

struct NetImpl: public torch::nn::Module {
    NetImpl() {
        // Initialize the network
        // On how to pass strides and padding: https://github.com/pytorch/pytorch/issues/12649#issuecomment-430156160
        conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 3).padding(1)));
        conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 3).padding(1)));
        // Insert pool layer
        conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 30, 3).padding(1)));
        conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(30, 40, 3).padding(1)));
        // Insert pool layer
        conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(40, 50, 3).padding(1)));
        conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(50, 60, 3).padding(1)));
        conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(60, 70, 3).padding(1)));
        // Insert pool layer
        conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(70, 80, 3).padding(1)));
        conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(80, 90, 3).padding(1)));
        conv4_3 = register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(90, 100, 3).padding(1)));
        // Insert pool layer
        conv5_1 = register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(100, 110, 3).padding(1)));
        conv5_2 = register_module("conv5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(110, 120, 3).padding(1)));
        conv5_3 = register_module("conv5_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(120, 130, 3).padding(1)));
        // Insert pool layer
        fc1 = register_module("fc1", torch::nn::Linear(130*6*6, 2000));
        fc2 = register_module("fc2", torch::nn::Linear(2000, 1000));
        fc3 = register_module("fc3", torch::nn::Linear(1000, 100));
        fc4 = register_module("fc4", torch::nn::Linear(100, 2));
    }

    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1_1->forward(x));
        x = torch::relu(conv1_2->forward(x));
        x = torch::max_pool2d(x, 2);

        x = torch::relu(conv2_1->forward(x));
        x = torch::relu(conv2_2->forward(x));
        x = torch::max_pool2d(x, 2);

        x = torch::relu(conv3_1->forward(x));
        x = torch::relu(conv3_2->forward(x));
        x = torch::relu(conv3_3->forward(x));
        x = torch::max_pool2d(x, 2);

        x = torch::relu(conv4_1->forward(x));
        x = torch::relu(conv4_2->forward(x));
        x = torch::relu(conv4_3->forward(x));
        x = torch::max_pool2d(x, 2);

        x = torch::relu(conv5_1->forward(x));
        x = torch::relu(conv5_2->forward(x));
        x = torch::relu(conv5_3->forward(x));
        x = torch::max_pool2d(x, 2);


        x = x.view({-1, 130*6*6});

        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));
        x = fc4->forward(x);
        return torch::log_softmax(x, 1);
    }

    // Declare layers
    torch::nn::Conv2d conv1_1{nullptr};
    torch::nn::Conv2d conv1_2{nullptr};
    torch::nn::Conv2d conv2_1{nullptr};
    torch::nn::Conv2d conv2_2{nullptr};
    torch::nn::Conv2d conv3_1{nullptr};
    torch::nn::Conv2d conv3_2{nullptr};
    torch::nn::Conv2d conv3_3{nullptr};
    torch::nn::Conv2d conv4_1{nullptr};
    torch::nn::Conv2d conv4_2{nullptr};
    torch::nn::Conv2d conv4_3{nullptr};
    torch::nn::Conv2d conv5_1{nullptr};
    torch::nn::Conv2d conv5_2{nullptr};
    torch::nn::Conv2d conv5_3{nullptr};

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
};

int main(int argc, char const *argv[]) {
	// Load the model.
	// Read Data
	vector<string> folders_name;
	folders_name.push_back("/media/cruigo/stuff/datasets/dogscats/train/cat");
	folders_name.push_back("/media/cruigo/stuff/datasets/dogscats/train/dog");

	std::pair<vector<string>, vector<int>> pair_images_labels = load_data_from_folder(folders_name);

	vector<string> list_images = pair_images_labels.first;
	vector<int> list_labels = pair_images_labels.second;

	auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());

	auto net = std::make_shared<NetImpl>();
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-3));

	int dataset_size = custom_dataset.size().value();
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset), 4);
	int n_epochs = 10; // Number of epochs
	float batch_index = 0;
	for(int epoch=1; epoch<=n_epochs; epoch++) {
		for(auto& batch: *data_loader) {
			auto data = batch.data;
			auto target = batch.target.squeeze();

			// Convert data to float32 format and target to Int64 format
			// Assuming you have labels as integers
			data = data.to(torch::kF32);
			target = target.to(torch::kInt64);

			// Clear the optimizer parameters
			optimizer.zero_grad();

			auto output = net->forward(data);
			auto loss = torch::nll_loss(output, target);

			// Backpropagate the loss
			loss.backward();
			// Update the parameters
			optimizer.step();

			cout << "Train Epoch: " << epoch << "Loss: " << loss.item<float>() << endl;
		}
	}

	// Save the model
	torch::save(net, "best_model.pt");
}