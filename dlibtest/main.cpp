#include "Header.h"

using namespace std;
using namespace dlib;

bool ListFiles(wstring path, wstring mask, std::vector<wstring>& files);

int main(int argc, char** argv)
{
	typedef matrix<double, 0, 1> sample_type;

	typedef radial_basis_kernel<sample_type> kernel_type;

	typedef std::vector<sample_type> sample_set;

	std::vector<sample_type> samples, tmp;
	std::vector<double> labels;
	std::vector<sample_set> data;
	
	wchar_t buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	wstring ws(buffer);
	string path(ws.begin(), ws.end());
	path = path.substr(0, path.find_last_of("\\/") + 1);
	cout << path << endl;

	std::vector<wstring> files;
	const size_t size = strlen(argv[1]) + 1;
	wchar_t* path_source = new wchar_t[size];
	mbstowcs(path_source, argv[1], size);

	cout << argv[1] << endl;

	if (ListFiles(path_source, L"*", files))
	{
		for (std::vector<wstring>::iterator it = files.begin();it != files.end(); ++it)
		{
			string path_file(it->begin(), it->end());
			cout << path_file << endl;
			deserialize(path_file) >> tmp;
			data.push_back(tmp);
		}
	}
	
	for (int i = 0; i < data.size(); i++)
	{
		for (int j = i + 1; j < data.size(); j++)
		{
			std::vector<double> labels;
			std::vector<sample_type> samples;
			samples.insert(samples.end(), data[i].begin(), data[i].end());
			samples.insert(samples.end(), data[j].begin(), data[j].end());

			for (int k = 0; k < data[i].size(); k++)
				labels.push_back(-1);
			for (int k = 0; k < data[j].size(); k++)
				labels.push_back(1);
			cout << "doing cross validation" << endl;
			vector_normalizer<sample_type> normalizer;
			normalizer.train(samples);
			for (unsigned long k = 0; k < samples.size(); ++k)
				samples[k] = normalizer(samples[k]);

			randomize_samples(samples, labels);
			const double max_nu = maximum_nu(labels);
			svm_nu_trainer<kernel_type> trainer;

			cout << "doing cross validation" << endl;
			double gamma, nu, max = 0, m_gamma, m_nu;
			for (gamma = 0.00001; gamma <= 1; gamma *= 5)
			{
				for (nu = 0.00001; nu < max_nu; nu *= 5)
				{
					// tell the trainer the parameters we want to use
					trainer.set_kernel(kernel_type(gamma));
					trainer.set_nu(nu);

					cout << "gamma: " << gamma << "    nu: " << nu << endl;

					matrix<double, 1, 2> res = cross_validate_trainer(trainer, samples, labels, 5);
					cout << "     cross validation accuracy: " << res(0) << " " << res(1) << endl;
					if (res(0) + res(1) > max)
					{
						max = res(0) + res(1);
						m_gamma = gamma;
						m_nu = nu;
						if (max >= 2)
							goto find;
					}
				}
			}
		find:
			cout << "max " << m_gamma << " " << m_nu << endl;
			trainer.set_kernel(kernel_type(m_gamma));
			trainer.set_nu(m_nu);
			typedef decision_function<kernel_type> dec_funct_type;
			typedef normalized_function<dec_funct_type> funct_type;

			funct_type learned_function;
			learned_function.normalizer = normalizer;  // save normalization information
			learned_function.function = trainer.train(samples, labels);

			cout << "\nnumber of support vectors in our learned_function is "<< learned_function.function.basis_vectors.size() << endl;
			typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
			typedef normalized_function<probabilistic_funct_type> pfunct_type;

			pfunct_type learned_pfunct;
			
		}
	}

	/*
	randomize_samples(samples, labels);

	const double max_nu = maximum_nu(labels);

	// here we make an instance of the svm_nu_trainer object that uses our kernel type.
	svm_nu_trainer<kernel_type> trainer;


	cout << "doing cross validation" << endl;
	for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
	{
		for (double nu = 0.00001; nu < max_nu; nu *= 5)
		{
			// tell the trainer the parameters we want to use
			trainer.set_kernel(kernel_type(gamma));
			trainer.set_nu(nu);

			cout << "gamma: " << gamma << "    nu: " << nu;
			
			matrix<double, 1 , 2> res = cross_validate_trainer(trainer, samples, labels, 5);
			cout << "     cross validation accuracy: " << res(0) << " " << res(1) << endl;
		}
	}

	trainer.set_kernel(kernel_type(0.00625));
	trainer.set_nu(0.00625);
	typedef decision_function<kernel_type> dec_funct_type;
	typedef normalized_function<dec_funct_type> funct_type;

	funct_type learned_function;
	learned_function.normalizer = normalizer;  // save normalization information
	learned_function.function = trainer.train(samples, labels); // perform the actual SVM training and save the results

																// print out the number of support vectors in the resulting decision function
	cout << "\nnumber of support vectors in our learned_function is "
		<< learned_function.function.basis_vectors.size() << endl;

	deserialize(path + "des.dat") >> tmp;
	// Now let's try this decision_function on some samples we haven't seen before.
	
	cout << "This is a +1 class example, the classifier output is " << learned_function(tmp[0]) << endl;

	typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
	typedef normalized_function<probabilistic_funct_type> pfunct_type;

	pfunct_type learned_pfunct;
	learned_pfunct.normalizer = normalizer;
	learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);
	
	cout << "\nnumber of support vectors in our learned_pfunct is "
		<< learned_pfunct.function.decision_funct.basis_vectors.size() << endl;

	serialize("saved_function.dat") << learned_pfunct;

	// Now let's open that file back up and load the function object it contains.
	deserialize("saved_function.dat") >> learned_pfunct;

	cout << "\ncross validation accuracy with only 10 support vectors: "
		<< cross_validate_trainer(reduced2(trainer, 10), samples, labels, 3);

	// Let's print out the original cross validation score too for comparison.
	cout << "cross validation accuracy with all the original support vectors: "
		<< cross_validate_trainer(trainer, samples, labels, 3);

	learned_function.function = reduced2(trainer, 10).train(samples, labels);
	// And similarly for the probabilistic_decision_function: 
	learned_pfunct.function = train_probabilistic_decision_function(reduced2(trainer, 10), samples, labels, 3);
	*/
	
}

bool ListFiles(wstring path, wstring mask, std::vector<wstring>& files) {
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA ffd;
	wstring spec;
	stack<wstring> directories;

	directories.push(path);
	files.clear();

	while (!directories.empty()) {
		path = directories.top();
		spec = path + L"/" + mask;
		directories.pop();

		hFind = FindFirstFile(spec.c_str(), &ffd);
		if (hFind == INVALID_HANDLE_VALUE) {
			return false;
		}

		do {
			if (wcscmp(ffd.cFileName, L".") != 0 &&
				wcscmp(ffd.cFileName, L"..") != 0) {
				if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
					directories.push(path + L"/" + ffd.cFileName);
				}
				else {
					files.push_back(path + L"/" + ffd.cFileName);
				}
			}
		} while (FindNextFile(hFind, &ffd) != 0);

		if (GetLastError() != ERROR_NO_MORE_FILES) {
			FindClose(hFind);
			return false;
		}

		FindClose(hFind);
		hFind = INVALID_HANDLE_VALUE;
	}

	return true;
}

/*
#include <dlib/svm_threaded.h>

#include <iostream>
#include <vector>

#include <dlib/rand.h>

using namespace std;
using namespace dlib;

// Our data will be 2-dimensional data. So declare an appropriate type to contain these points.
typedef matrix<double, 0, 1> sample_type;

// ----------------------------------------------------------------------------------------

void generate_data(
	std::vector<sample_type>& samples,
	std::vector<double>& labels
);

// ----------------------------------------------------------------------------------------

int main()
{
	try
	{


		std::vector<sample_type> samples, fileA, fileB, fileC, fileD, test;
		std::vector<double> labels;

		wchar_t buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		wstring ws(buffer);
		string path(ws.begin(), ws.end());
		path = path.substr(0, path.find_last_of("\\/") + 1);


		deserialize(path + "m.01b5zn.dat") >> fileA;
		deserialize(path + "m.01b_88.dat") >> fileB;
		deserialize(path + "m.01b0fq.dat") >> fileC;
		deserialize(path + "trump.dat") >> fileD;

		for (int i = 0; i < fileA.size(); i++)
			labels.push_back(1);
		for (int i = 0; i < fileB.size(); i++)
			labels.push_back(2);
		for (int i = 0; i < fileC.size(); i++)
			labels.push_back(3);
		for (int i = 0; i < fileD.size(); i++)
			labels.push_back(4);

		samples.insert(samples.end(), fileA.begin(), fileA.end());
		samples.insert(samples.end(), fileB.begin(), fileB.end());
		samples.insert(samples.end(), fileC.begin(), fileC.end());
		samples.insert(samples.end(), fileD.begin(), fileD.end());

		typedef one_vs_all_trainer<any_trainer<sample_type> > ova_trainer;

		// Finally, make the one_vs_one_trainer.
		ova_trainer trainer;

		typedef polynomial_kernel<sample_type> poly_kernel;
		typedef radial_basis_kernel<sample_type> rbf_kernel;

		// make the binary trainers and set some parameters
		krr_trainer<rbf_kernel> rbf_trainer;
		svm_nu_trainer<poly_kernel> poly_trainer;
		poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
		rbf_trainer.set_kernel(rbf_kernel(0.1));
		
		trainer.set_trainer(rbf_trainer);
		
		trainer.set_trainer(poly_trainer);

		
		randomize_samples(samples, labels);
		for (int i = 0; i < labels.size(); i++)
			cout << labels[i] << endl;

		cout << "size -->" << labels.size() << endl;
		cout << "A -->" << fileA.size() << endl;
		cout << "B -->" << fileB.size() << endl;
		cout << "C -->" << fileC.size() << endl;
		cout << "D -->" << fileD.size() << endl;

		cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
		
		one_vs_all_decision_function<ova_trainer> df = trainer.train(samples, labels);

		cout << "predicted label: " << df(samples[0]) << ", true label: " << labels[0] << endl;
		cout << "predicted label: " << df(samples[6]) << ", true label: " << labels[6] << endl;
		

		one_vs_all_decision_function<ova_trainer,
			decision_function<poly_kernel>  // This is the output of the poly_trainer
		> df2, df3;


		df2 = df;
		serialize(path + "df.dat") << df2;

		// load the function back in from disk and store it in df3.  
		deserialize(path + "df.dat") >> df3;
		
		deserialize(path + "des.dat") >> test;

		// Test df3 to see that this worked.
		cout << endl;
		cout << "predicted label: " << df3(test[2]) << ", true label: 1" << endl;
		cout << "predicted label: " << df3(test[1]) << ", true label: 1"  << endl;
		cout << "predicted label: " << df3(test[3]) << ", true label: 2" << endl;
		cout << "predicted label: " << df3(test[5]) << ", true label: 2" << endl;
		cout << "predicted label: " << df3(test[6]) << ", true label: 3" << endl;
		cout << "predicted label: " << df3(test[8]) << ", true label: 3" << endl;
		cout << "predicted label: " << df3(test[9]) << ", true label: 4" << endl;
		cout << "predicted label: " << df3(test[10]) << ", true label: 4" << endl;

		one_vs_all_decision_function<ova_trainer>::binary_function_table tmp;
		tmp = df3.get_binary_decision_functions();
		cout << tmp.size() << endl;
		
		cout << "test deserialized function: \n" << test_multiclass_decision_function(df3, samples, labels) << endl;

		
		one_vs_all_decision_function<ova_trainer>::binary_function_table functs;
		functs = df.get_binary_decision_functions();
		cout << "number of binary decision functions in df: " << functs.size() << endl;
		
		//decision_function<poly_kernel> df_1_2 = any_cast<decision_function<poly_kernel> >(functs[make_unordered_pair(1, 2)]);
		
	
	}
	catch (std::exception& e)
	{
		cout << "exception thrown!" << endl;
		cout << e.what() << endl;
	}
}

// ----------------------------------------------------------------------------------------

void generate_data(
	std::vector<sample_type>& samples,
	std::vector<double>& labels
)
{
	const long num = 50;

	sample_type m;

	dlib::rand rnd;


	// make some samples near the origin
	double radius = 0.5;
	for (long i = 0; i < num + 10; ++i)
	{
		double sign = 1;
		if (rnd.get_random_double() < 0.5)
			sign = -1;
		m(0) = 2 * radius*rnd.get_random_double() - radius;
		m(1) = sign*sqrt(radius*radius - m(0)*m(0));

		// add this sample to our set of training samples 
		samples.push_back(m);
		labels.push_back(1);
	}

	// make some samples in a circle around the origin but far away
	radius = 10.0;
	for (long i = 0; i < num + 20; ++i)
	{
		double sign = 1;
		if (rnd.get_random_double() < 0.5)
			sign = -1;
		m(0) = 2 * radius*rnd.get_random_double() - radius;
		m(1) = sign*sqrt(radius*radius - m(0)*m(0));

		// add this sample to our set of training samples 
		samples.push_back(m);
		labels.push_back(2);
	}

	// make some samples in a circle around the point (25,25) 
	radius = 4.0;
	for (long i = 0; i < num + 30; ++i)
	{
		double sign = 1;
		if (rnd.get_random_double() < 0.5)
			sign = -1;
		m(0) = 2 * radius*rnd.get_random_double() - radius;
		m(1) = sign*sqrt(radius*radius - m(0)*m(0));

		// translate this point away from the origin
		m(0) += 25;
		m(1) += 25;

		// add this sample to our set of training samples 
		samples.push_back(m);
		labels.push_back(3);
	}
}

// ----------------------------------------------------------------------------------------

*/