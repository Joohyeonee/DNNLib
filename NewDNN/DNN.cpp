#include "DNN.h"

void setInput(NeuralNet &NN)
{
	int nRow = NN.data.size();
	int nCol = NN.nNodes[0];
	for (int i = 0; i < nRow; i++)
	{
		for (int j = 0; j < nCol; j++)
		{
			NN.input[i][j] = NN.data[i][j];
		}
	}
}

void setInput(NeuralNet &NN, int startRow, int EndRow)
{
	int nRow = 0;
	int nCol = NN.nNodes[0];
	for (int i = startRow; i < EndRow; i++)
	{
		for (int j = 0; j < nCol; j++)
		{
			NN.input[nRow][j] = NN.data[i][j];
		}
		nRow++;
	}
}

void setTarget(NeuralNet &NN)
{
	int nRow = NN.data.size();
	int nCol = NN.nNodes[NN.nNodes.size() - 1];
	if (NN.learn_method == prediction)
	{
		for (int i = 0; i < nRow; i++)
		{
			for (int j = 0; j < nCol; j++)
			{
				NN.target[i][nCol - 1] = NN.data[i][NN.nNodes[0]];
			}
		}
	}
	else if (NN.learn_method == classification)
	{
		for (int i = 0; i < nRow; i++)
		{
			int targetCol = NN.data[i][NN.nNodes[0]];
			NN.target[i][targetCol] = 1;
		}
	}
}

void setTarget(NeuralNet &NN, int startRow, int EndRow)
{
	int nRow = 0;
	int nCol = NN.nNodes[NN.nNodes.size() - 1];
	if (NN.learn_method == prediction)
	{	
		for (int i = startRow; i < EndRow; i++)
		{
			for (int j = 0; j < nCol; j++)
			{
				NN.target[nRow][nCol - 1] = NN.data[i][NN.nNodes[0]];
			}
			nRow++;
		}
	}
	else if (NN.learn_method == classification)
	{
		for (int i = startRow; i < EndRow; i++)
		{
			int targetCol = NN.data[i][NN.nNodes[0]];
			for (int j = 0; j < nCol; j++)
			{
				if (j == targetCol)NN.target[nRow][j] = 1;
				else NN.target[nRow][j] = 0;
			}
			nRow++;
		}
	}
	
}

void normalization(NeuralNet &NN)
{
	int numrows = 0;
	int numcols = 0;
	if (NN.learn_method == prediction)
	{
		numrows = NN.data.size();
		numcols = NN.data[0].size();
	}
	else if (NN.learn_method == classification)
	{
		numrows = NN.data.size();
		numcols = NN.data[0].size() - 1;
	}
	if (NN.nEpoch != -1)
	{
		//1.Min-Max Scaling 
		if (NN.scaler == minmax) {
			for (int i = 0; i < numcols; i++) {
				double maxTmp = NN.data[0][i];
				double minTmp = NN.data[0][i];
				for (int j = 0; j < numrows; j++) {
					if (NN.data[j][i] > maxTmp) {
						maxTmp = NN.data[j][i];
					}
					if (NN.data[j][i] < minTmp) {
						minTmp = NN.data[j][i];
					}
				}
				for (int j = 0; j < numrows; j++) {
					NN.data[j][i] = (NN.data[j][i] - minTmp) / (maxTmp - minTmp);
				}
				NN.max[i] = maxTmp;
				NN.min[i] = minTmp;
			}
		}
		//2.Max Abs Normalization
		else if (NN.scaler == maxabs) {
			for (int i = 0; i < numcols; i++) {
				double maxTmp = abs(NN.data[0][i]);
				for (int j = 0; j < numrows; j++) {
					if (abs(NN.data[j][i]) > abs(maxTmp)) {
						maxTmp = abs(NN.data[j][i]);
					}
				}
				for (int j = 0; j < numrows; j++) {
					NN.data[j][i] = NN.data[j][i] / maxTmp;
				}
				NN.max[i] = maxTmp;
			}
		}
		//3.Standardization
		else if (NN.scaler == standard) {
			for (int i = 0; i < numcols; i++) {
				double meanTmp = 0;
				double sum = 0;
				double sum2 = 0;
				for (int j = 0; j < numrows; j++) {
					sum += NN.data[j][i];
				}
				meanTmp = sum / numrows;
				for (int j = 0; j < numrows; j++) {
					sum2 += pow((NN.data[j][i] - meanTmp), 2);
				}
				double var = sum2 / numrows;
				double stdTmp = sqrt(var);
				for (int j = 0; j < numrows; j++) {
					NN.data[j][i] = (NN.data[j][i] - meanTmp) / stdTmp;
				}
				NN.mean[i] = meanTmp;
				NN.std[i] = stdTmp;
			}
		}
	}
	else
	{
		//1.Min-Max Scaling 
		if (NN.scaler == minmax) {
			for (int i = 0; i < numcols; i++) {
				for (int j = 0; j < numrows; j++) {
					NN.data[j][i] = (NN.data[j][i] - NN.min[i]) / (NN.max[i] - NN.min[i]);
				}
			}
		}
		//2.Max Abs Normalization
		else if (NN.scaler == maxabs) {
			for (int i = 0; i < numcols; i++) {
				for (int j = 0; j < numrows; j++) {
					NN.data[j][i] = NN.data[j][i] / NN.max[i];
				}
			}
		}
		//3.Standardization
		else if (NN.scaler == standard) {
			for (int i = 0; i < numcols; i++) {
				for (int j = 0; j < numrows; j++) {
					NN.data[j][i] = (NN.data[j][i] - NN.mean[i]) / NN.std[i];
				}
			}
		}
	}
}

void denormalization(NeuralNet &NN)
{
	int t_Nrows = 0;
	int t_Ncols = 0;
	if (NN.learn_method == prediction)
	{
		t_Nrows = NN.data.size();
		t_Ncols = NN.data[0].size();
	}
	else if (NN.learn_method == classification)
	{
		t_Nrows = NN.data.size();
		t_Ncols = NN.data[0].size() - 1;
	}
	if (NN.scaler == minmax) {
		for (int i = 0; i < t_Ncols; i++) {
			for (int j = 0; j < t_Nrows; j++) {
				NN.data[j][i] = ((NN.max[i] - NN.min[i]) * NN.data[j][i]) + NN.min[i];
			}
		}
	}
	else if (NN.scaler == maxabs) {
		for (int i = 0; i < t_Ncols; i++) {
			for (int j = 0; j < t_Nrows; j++) {
				NN.data[j][i] = NN.max[i] * NN.data[j][i];
			}
		}
	}
	else if (NN.scaler == standard) {
		for (int i = 0; i < t_Ncols; i++) {
			for (int j = 0; j < t_Nrows; j++) {
				NN.data[j][i] = (NN.std[i] * NN.data[j][i]) + NN.mean[i];
			}
		}
	}
}

//feed forward
void feedForward(NeuralNet &NN) 
{
	int hiddenLayer = NN.nHidden;
	for (int layer = 0; layer < hiddenLayer + 1; layer++) 
	{ 
		int nRow = NN.nNodes[layer];
		int nCol = NN.nNodes[layer + 1];
		if (layer == 0) //Input Layer
		{ 
			for (int i = 0; i < NN.nBatch; i++)
			{
				for (int j = 0; j < nCol; j++) 
				{
					double sum = 0;
					for (int k = 0; k < nRow; k++)sum += NN.input[i][k] * NN.weight[layer][k][j];
					NN.hidden[layer][i][j] = sum + NN.bias[layer][j];
					NN.act_hidden[layer][i][j] = activeF(NN.hidden[layer][i][j], NN.actf_hidden, 0);
				}
			}
		}
		else if (layer < hiddenLayer) //Hidden Layer
		{ 
			for (int i = 0; i < NN.nBatch; i++) 
			{
				
				for (int j = 0; j < nCol; j++) 
				{
					double sum = 0;
					for (int k = 0; k < nRow; k++)sum += NN.act_hidden[layer - 1][i][k] * NN.weight[layer][k][j];
					NN.hidden[layer][i][j] = sum + NN.bias[layer][j];
					NN.act_hidden[layer][i][j] = activeF(NN.hidden[layer][i][j], NN.actf_hidden, 0);
				}
			}
		}
		else //Output Layer
		{
			for (int i = 0; i < NN.nBatch; i++)
			{	
				for (int j = 0; j < nCol; j++)
				{
					double sum = 0;
					for (int k = 0; k < nRow; k++)sum += NN.act_hidden[layer - 1][i][k] * NN.weight[layer][k][j];
					NN.output[i][j] = sum + NN.bias[layer][j];
					double softmaxsum = 0;
					double softmaxMax = 0;
					if (NN.actf_output == softmax)
					{
						for (int idx = 0; idx < nCol; idx++)
						{
							if (softmaxMax < NN.output[i][idx]) softmaxMax = NN.output[i][idx];
						}
						for (int idx = 0; idx < nCol; idx++)softmaxsum += exp(NN.output[i][idx] - softmaxMax);
					}
					NN.act_output[i][j] = activeF(NN.output[i][j] - softmaxMax, NN.actf_output, softmaxsum);
				}
			}
		}
	}
}

//back propagation
void backPropagation(NeuralNet &NN) {
	int hiddenLayer = NN.nHidden;
	for (int layer = hiddenLayer; layer >= 0; layer--) {
		int nRow = NN.nNodes[layer];
		int nCol = NN.nNodes[layer + 1];
		if (layer == hiddenLayer) { // 출력층에서 은닉층으로 가는 Error 계산
			for (int i = 0; i < NN.nBatch; i++) {
				for (int j = 0; j < nCol; j++) {
					NN.error[layer][i][j] = (NN.act_output[i][j] - NN.target[i][j]) * dactiveF(NN.output[i][j], NN.actf_output, 0);
				}
			}
		}
		else { // 은닉층에서 은닉층으로 가는 Error 계산
			for (int i = 0; i < NN.nBatch; i++) {
				for (int j = 0; j < nCol; j++) {
					NN.error[layer][i][j] = NN.error_in[layer][i][j] * dactiveF(NN.hidden[layer][i][j], NN.actf_hidden, 0);
				}
			}
		}
		switch (NN.optimizer)
		{
		case gdm: GDMoment(NN, layer);
			break;
		case rmsprop: RMSProp(NN, layer);
			break;
		case adagrad: Adagrad(NN, layer);
			break;
		case adam: Adam(NN, layer);
			break;
		}
		// 이전층에서 다음층으로 갈 때 오차정보 계산
		if (layer != 0) {
			for (int i = 0; i < NN.nBatch; i++) 
			{
				for (int j = 0; j < nRow; j++)
				{
					double sum = 0;
					for (int k = 0; k < nCol; k++)sum += NN.error[layer][i][k] * NN.weight[layer][j][k];
					NN.error_in[layer - 1][i][j] = sum;
				}
			}
		}
	}
}

void predict(NeuralNet &NN)
{
	int hiddenLayer = NN.nHidden;
	for (int layer = 0; layer < hiddenLayer + 1; layer++)
	{
		int nRow = NN.nNodes[layer];
		int nCol = NN.nNodes[layer + 1];
		if (layer == 0) //Input Layer
		{
			for (int i = 0; i < NN.nBatch; i++)
			{

				for (int j = 0; j < nCol; j++)
				{
					double sum = 0;
					for (int k = 0; k < nRow; k++)sum += NN.input[i][k] * NN.weight[layer][k][j];
					NN.hidden[layer][i][j] = sum + NN.bias[layer][j];
					NN.act_hidden[layer][i][j] = activeF(NN.hidden[layer][i][j], NN.actf_hidden, 0);
				}
			}
		}
		else if (layer < hiddenLayer) //Hidden Layer
		{
			for (int i = 0; i < NN.nBatch; i++)
			{

				for (int j = 0; j < nCol; j++)
				{
					double sum = 0;
					for (int k = 0; k < nRow; k++)sum += NN.act_hidden[layer - 1][i][k] * NN.weight[layer][k][j];
					NN.hidden[layer][i][j] = sum + NN.bias[layer][j];
					NN.act_hidden[layer][i][j] = activeF(NN.hidden[layer][i][j], NN.actf_hidden, 0);
				}
			}
		}
		else //Output Layer
		{
			for (int i = 0; i < NN.nBatch; i++)
			{
				double max = 0;
				for (int j = 0; j < nCol; j++)
				{
					double sum = 0;
					for (int k = 0; k < nRow; k++)sum += NN.act_hidden[layer - 1][i][k] * NN.weight[layer][k][j];
					NN.output[i][j] = sum + NN.bias[layer][j];
					double softmaxsum = 0;
					double softmaxMax = 0;

					if (NN.actf_output == softmax)
					{
						for (int idx = 0; idx < nCol; idx++)
						{
							if (softmaxMax < NN.output[i][idx]) softmaxMax = NN.output[i][idx];
						}
						for (int idx = 0; idx < nCol; idx++)softmaxsum += exp(NN.output[i][idx] - softmaxMax);
					}
					NN.act_output[i][j] = activeF(NN.output[i][j] - softmaxMax, NN.actf_output, softmaxsum);
					int outputCol = NN.data[0].size() - 1;

					if (NN.learn_method == prediction)
					{
						NN.data[i][outputCol] = NN.act_output[i][j];
					}
					else if (NN.learn_method == classification)
					{
						if (max < NN.act_output[i][j])
						{
							max = NN.act_output[i][j];
							NN.data[i][outputCol] = j;
						}
					}
				}
			}
		}
	}
}

//Activation Functions
double linearF(double x) {
	return x;
}

double dlinearF(double x) {
	return 1;
}


double sigmoidF(double x) {
	x = 1 / (1 + exp(-x));
	return x;
}

double dsigmoidF(double x) {
	x = (1 / (1 + exp(-x))) * (1 - 1 / (1 + exp(-x)));
	return x;
}

double reluF(double x)
{
	if (x > 0) return x;
	else return 0;
}

double dreluF(double x)
{
	if (x > 0) return 1;
	else return 0;
}

double softMaxF(double x, double sum)
{
	return exp(x) / sum;
}

double dsoftMaxF(double x)
{
	return 1;
}

double activeF(double x, int act, double sum)
{
	if (act == sigmoid) return sigmoidF(x);
	else if (act == relu) return reluF(x);
	else if (act == linear) return linearF(x);
	else if (act == softmax) return softMaxF(x, sum);
}

double dactiveF(double x, int act, double sum)
{
	if (act == sigmoid) return dsigmoidF(x);
	else if (act == relu) return dreluF(x);
	else if (act == linear) return dlinearF(x);
	else if (act == softmax) return dsoftMaxF(x);
}

//update weight
void updateWeight(NeuralNet &NN) {
	int hiddenLayer = NN.nHidden;
	for (int layer = 0; layer < hiddenLayer + 1; layer++) 
	{
		int nRow = NN.nNodes[layer];
		int nCol = NN.nNodes[layer + 1];
		int batch = NN.nBatch;
		for (int i = 0; i < nRow; i++) 
		{
			for (int j = 0; j < nCol; j++) 
			{
				NN.weight[layer][i][j] -= NN.dweight[layer][i][j];
				NN.dweight[layer][i][j] = 0;
			}
		}
		for (int i = 0; i < batch; i++) 
		{
			for (int j = 0; j < nCol; j++) 
			{
				NN.bias[layer][j] -= NN.dbias[layer][i][j]/batch;
				NN.dbias[layer][i][j] = 0;
			}
		}
	}
}

void makeModel(NeuralNet &model, NeuralNet &NN)
{
	model.nNodes = NN.nNodes;
	model.nHidden = NN.nHidden;
	model.scaler = NN.scaler;
	model.optimizer = NN.optimizer;
	model.actf_hidden = NN.actf_hidden;
	model.actf_output = NN.actf_output;
	model.learn_method = NN.learn_method;

	model.weight = new double**[model.nHidden + 1];
	model.bias = new double*[model.nHidden + 1];
	for (int i = 0; i < model.nHidden + 1; i++)
	{
		model.weight[i] = new double*[model.nNodes[i]];
		for (int j = 0; j < model.nNodes[i]; j++)
		{
			model.weight[i][j] = new double[model.nNodes[i + 1]]();
			for (int k = 0; k < model.nNodes[i + 1]; k++)
			{
				model.weight[i][j][k] = NN.weight[i][j][k];
			}
		}
		model.bias[i] = new double[model.nNodes[i + 1]]();
		for (int k = 0; k < model.nNodes[i + 1]; k++)
		{
			model.bias[i][k] = NN.bias[i][k];
		}
	}

	model.max = NN.max;
	model.min = NN.min;
	model.mean = NN.mean;
	model.std = NN.std;

	model.nBatch = model.data.size();
	model.nEpoch = -1;
	model.initLayer();
}

//optimizer
void GDMoment(NeuralNet &NN, int layer) 
{
	int nRow = NN.nNodes[layer];
	int nCol = NN.nNodes[layer + 1];
	double *g = new double[nRow*nCol]();
	if (layer == 0)
	{
		for (int i = 0; i < nRow; i++)
		{

			for (int j = 0; j < nCol; j++)
			{
				double sum = 0;
				for (int k = 0; k < NN.nBatch; k++)sum += NN.input[k][i] * NN.error[layer][k][j];
				g[i*nCol + j] = sum / NN.nBatch;
			}
		}
	}
	else {
		for (int i = 0; i < nRow; i++)
		{
			for (int j = 0; j < nCol; j++)
			{
				double sum = 0;
				for (int k = 0; k < NN.nBatch; k++)sum += NN.act_hidden[layer - 1][k][i] * NN.error[layer][k][j];
				g[i*nCol + j] = sum / NN.nBatch;
			}
		}
	}
	for (int i = 0; i < nRow; i++) 
	{
		for (int j = 0; j < nCol; j++) 
		{
			NN.opt_weight[layer][i][j] = NN.momentum * NN.opt_weight[layer][i][j] + NN.learningRate * g[i*nCol + j];
			NN.dweight[layer][i][j] = NN.opt_weight[layer][i][j];
		}
	}
	for (int i = 0; i < NN.nBatch; i++) {
		for (int j = 0; j < nCol; j++) {
			NN.opt_bias[layer][i][j] = NN.momentum * NN.opt_bias[layer][i][j] + NN.learningRate * NN.error[layer][i][j];
			NN.dbias[layer][i][j] = NN.opt_bias[layer][i][j];
		}
	}
	delete[] g;
}

void RMSProp(NeuralNet &NN, int layer) {
	int nRow = NN.nNodes[layer];
	int nCol = NN.nNodes[layer + 1];
	double *g = new double[nRow*nCol]();
	if (layer == 0)
	{
		for (int i = 0; i < nRow; i++)
		{

			for (int j = 0; j < nCol; j++)
			{
				double sum = 0;
				for (int k = 0; k < NN.nBatch; k++)sum += NN.input[k][i] * NN.error[layer][k][j];
				g[i*nCol + j] = sum / NN.nBatch;
			}
		}
	}
	else {
		for (int i = 0; i < nRow; i++)
		{
			for (int j = 0; j < nCol; j++)
			{
				double sum = 0;
				for (int k = 0; k < NN.nBatch; k++)sum += NN.act_hidden[layer - 1][k][i] * NN.error[layer][k][j];
				g[i*nCol + j] = sum / NN.nBatch;
			}
		}
	}
	for (int i = 0; i < nRow; i++) {
		for (int j = 0; j < nCol; j++) {
			NN.opt_weight[layer][i][j] = NN.gamma * NN.opt_weight[layer][i][j] + (1 - NN.gamma) * (g[i*nCol + j] * g[i*nCol + j]);
			NN.dweight[layer][i][j] = NN.learningRate * g[i*nCol + j] / (sqrt(NN.opt_weight[layer][i][j] + 1e-8));
		}
	}
	for (int i = 0; i < NN.nBatch; i++) {
		for (int j = 0; j < nCol; j++) {
			NN.opt_bias[layer][i][j] = NN.gamma * NN.opt_bias[layer][i][j] + (1 - NN.gamma) * (NN.error[layer][i][j] * NN.error[layer][i][j]);
			NN.dbias[layer][i][j] = NN.learningRate * NN.error[layer][i][j] / (sqrt(NN.opt_bias[layer][i][j] + 1e-8));
		}
	}
	delete[] g;
}

void Adagrad(NeuralNet &NN, int layer) {
	int nRow = NN.nNodes[layer];
	int nCol = NN.nNodes[layer + 1];
	double *g = new double[nRow*nCol]();
	if (layer == 0)
	{
		for (int i = 0; i < nRow; i++)
		{

			for (int j = 0; j < nCol; j++)
			{
				double sum = 0;
				for (int k = 0; k < NN.nBatch; k++)sum += NN.input[k][i] * NN.error[layer][k][j];
				g[i*nCol + j] = sum / NN.nBatch;
			}
		}
	}
	else {
		for (int i = 0; i < nRow; i++)
		{
			for (int j = 0; j < nCol; j++)
			{
				double sum = 0;
				for (int k = 0; k < NN.nBatch; k++)sum += NN.act_hidden[layer - 1][k][i] * NN.error[layer][k][j];
				g[i*nCol + j] = sum / NN.nBatch;
			}
		}
	}
	for (int i = 0; i < nRow; i++) {
		for (int j = 0; j < nCol; j++) {
			NN.opt_weight[layer][i][j] = NN.opt_weight[layer][i][j] + (g[i*nCol + j] * g[i*nCol + j]);
			NN.dweight[layer][i][j] = NN.learningRate * g[i*nCol + j] / (sqrt(NN.opt_weight[layer][i][j] + 1e-8));
		}
	}
	for (int i = 0; i < NN.nBatch; i++) {
		for (int j = 0; j < nCol; j++) {
			NN.opt_bias[layer][i][j] = NN.opt_bias[layer][i][j] + (NN.error[layer][i][j] * NN.error[layer][i][j]);
			NN.dbias[layer][i][j] = NN.learningRate * NN.error[layer][i][j] / (sqrt(NN.opt_bias[layer][i][j] + 1e-8));
		}
	}
	delete[] g;
}

void Adam(NeuralNet &NN, int layer) 
{
	int nRow = NN.nNodes[layer];
	int nCol = NN.nNodes[layer + 1];
	double *g = new double[nRow*nCol]();
	if (layer == 0)
	{
		for (int i = 0; i < nRow; i++)
		{

			for (int j = 0; j < nCol; j++)
			{
				double sum = 0;
				for (int k = 0; k < NN.nBatch; k++)sum = (NN.input[k][i] * NN.error[layer][k][j]);
				g[i*nCol + j] = sum / NN.nBatch;
			}
		}
	}
	else {
		for (int i = 0; i < nRow; i++)
		{

			for (int j = 0; j < nCol; j++)
			{
				double sum = 0;
				for (int k = 0; k < NN.nBatch; k++)sum = (NN.act_hidden[layer - 1][k][i] * NN.error[layer][k][j]);
				g[i*nCol + j] = sum / NN.nBatch;
			}
		}
	}
	for (int i = 0; i < nRow; i++) {
		for (int j = 0; j < nCol; j++) {
			NN.opt_weight_adam[layer][i][j] = NN.beta1 * NN.opt_weight_adam[layer][i][j] + (1 - NN.beta1) * g[i*nCol + j];
			NN.opt_weight[layer][i][j] = NN.beta2 * NN.opt_weight[layer][i][j] + (1 - NN.beta2) * g[i*nCol + j] * g[i*nCol + j];
			NN.dweight[layer][i][j] = NN.opt_weight_adam[layer][i][j] * (NN.learningRate / sqrt(NN.opt_weight[layer][i][j] + 1e-8));
		}
	}
	for (int j = 0; j < nCol; j++) {
		for (int k = 0; k < NN.nBatch; k++) {
			NN.opt_bias_adam[layer][k][j] = NN.beta1 *NN.opt_bias_adam[layer][k][j] + (1 - NN.beta1) * NN.error[layer][k][j];
			NN.opt_bias[layer][k][j] = NN.beta2 * NN.opt_bias[layer][k][j] + (1 - NN.beta2) * (NN.error[layer][k][j] * NN.error[layer][k][j]);
			NN.dbias[layer][k][j] = NN.opt_bias_adam[layer][k][j] * (NN.learningRate / sqrt(NN.opt_bias[layer][k][j] + 1e-8));
		}
	}
	delete[] g;
}

double calcRMSE(NeuralNet &NN)
{
	int NRows = NN.nBatch;
	int NCols = NN.nNodes[NN.nNodes.size() - 1];
	int TotNum = NRows * NCols;

	double mse = 0;

	for (int ridx = 0; ridx < NRows; ridx++)
	{
		for (int cidx = 0; cidx < NCols; cidx++)
		{
			mse += (NN.act_output[ridx][cidx] - NN.target[ridx][cidx]) * (NN.act_output[ridx][cidx] - NN.target[ridx][cidx]);
		}
	}

	mse /= TotNum;
	return sqrt(mse);
}