#include"net_builder.h"
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>

using dataset = std::pair<images, labels>;
std::string datasets_root = "datasets\\";

dataset* read_csv(std::string csv_name, size_t rows);
float accuracy(const labels& expected, const labels& actual);

int main() {
    dataset* train = read_csv(datasets_root + "fashion_train.csv", 60000);
    convolutional_network cnn = net_builder().request(28, 28, 10).conv().relu().pool()
        .full_con().hidden(200).with_relu().last().with_relu().build();
	cnn.fit(train->first, train->second);

    dataset* test = read_csv(datasets_root + "fashion_test.csv", 10000);
    labels predicted = cnn.predict(test->first);
    std::cout << "--- Accuracy: " << accuracy(test->second, predicted) << std::endl;
    delete train, test;
    return 0;
}

dataset* read_csv(std::string csv_name, size_t rows) {
    std::cout << "--- Reading " << csv_name << " ---" << std::endl;
    std::ifstream csv_file(csv_name);
    csv_file.seekg(0, std::ios::end);
    auto csv_size = csv_file.tellg();
    csv_file.seekg(0);
    char* csv_content = new char[static_cast<size_t>(csv_size)];
    csv_file.read(csv_content, csv_size);

    std::stringstream csv_stream(csv_content);
    images input_images(rows, matrix({ 28, 28 }));
    labels input_labels(rows);
    std::string line, cell;
    std::getline(csv_stream, line);

    for (size_t row = 0; row < rows; ++row) {
        std::getline(csv_stream, line);
        std::stringstream line_stream(line);
        for (size_t pix = 0; pix < 784; ++pix) {
            std::getline(line_stream, cell, ',');
            input_images[row].kern[pix] = atoi(cell.c_str()) / 255.0f;
        }
        std::getline(line_stream, cell, ',');
        size_t label = static_cast<size_t>(atoi(cell.c_str()));
        input_labels[row] = label;
    }
    delete[] csv_content;
    std::cout << "--- Parsing " << csv_name << " is successful ---" << std::endl;
    return new dataset({ input_images, input_labels });
}

float accuracy(const labels& expected, const labels& actual) {
    size_t ok = 0;
    for (size_t sample = 0; sample < actual.size(); ++sample) {
        if (expected[sample] == actual[sample]) { ok++; }
    }
    return static_cast<float>(ok) / actual.size();
}