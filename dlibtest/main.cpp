

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
/*!
ensures
- make some 3 class data as described above.
- Create 60 points from class 1
- Create 70 points from class 2
- Create 80 points from class 3
!*/

// ----------------------------------------------------------------------------------------

int main()
{
	try
	{


		std::vector<sample_type> samples, fileA, fileB, fileC, test;
		std::vector<double> labels;

		wchar_t buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		wstring ws(buffer);
		string path(ws.begin(), ws.end());
		path = path.substr(0, path.find_last_of("\\/") + 1);


		deserialize(path + "m.01b5zn.dat") >> fileA;
		deserialize(path + "m.01b_88.dat") >> fileB;
		deserialize(path + "m.01b0fq.dat") >> fileC;

		for (int i = 0; i < fileA.size(); i++)
			labels.push_back(1);
		for (int i = 0; i < fileB.size(); i++)
			labels.push_back(2);
		for (int i = 0; i < fileC.size(); i++)
			labels.push_back(3);

		samples.insert(samples.end(), fileA.begin(), fileA.end());
		samples.insert(samples.end(), fileB.begin(), fileB.end());
		samples.insert(samples.end(), fileC.begin(), fileC.end());

		typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;

		// Finally, make the one_vs_one_trainer.
		ovo_trainer trainer;

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

		cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
		
		// Next, if you wanted to obtain the decision rule learned by a one_vs_one_trainer you 
		// would store it into a one_vs_one_decision_function.
		one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);

		cout << "predicted label: " << df(samples[0]) << ", true label: " << labels[0] << endl;
		cout << "predicted label: " << df(samples[6]) << ", true label: " << labels[6] << endl;
		

		one_vs_one_decision_function<ovo_trainer,
			decision_function<poly_kernel>,  // This is the output of the poly_trainer
			decision_function<rbf_kernel>    // This is the output of the rbf_trainer
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
		// Test df3 on the samples and labels and print the confusion matrix.
		cout << "test deserialized function: \n" << test_multiclass_decision_function(df3, samples, labels) << endl;

		// Finally, if you want to get the binary classifiers from inside a multiclass decision
		// function you can do it by calling get_binary_decision_functions() like so:
		one_vs_one_decision_function<ovo_trainer>::binary_function_table functs;
		functs = df.get_binary_decision_functions();
		cout << "number of binary decision functions in df: " << functs.size() << endl;
		// The functs object is a std::map which maps pairs of labels to binary decision
		// functions.  So we can access the individual decision functions like so:
		decision_function<poly_kernel> df_1_2 = any_cast<decision_function<poly_kernel> >(functs[make_unordered_pair(1, 2)]);
		//decision_function<rbf_kernel>  df_1_3 = any_cast<decision_function<rbf_kernel>  >(functs[make_unordered_pair(1, 3)]);
	
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

