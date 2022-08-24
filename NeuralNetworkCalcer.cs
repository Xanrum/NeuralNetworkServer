namespace NeuralNetworkServer;

public class NeuralNetworkCalcer
{
    public NeuralNetworkCalcer(int[] model)
    {
        _model = model;
    }
    private readonly int[] _model;


    public double[] Calc(double[] inputs, int startInput, double[] synapses)
    {
        var max = _model.Max();
        var prev = new double[max];
        Array.Copy(inputs, startInput, prev, 0, _model[0]);
        var next = new double[max];
        var sinapsIndex = 0;
        for (var i = 1; i < _model.Length; i++)
        {
            var len = _model[i];
            for (var i1 = 0; i1 < len; i1++)
            {
                var sum = 0d;
                foreach (var p in prev)
                {
                    sum += p * synapses[sinapsIndex];
                    sinapsIndex++;
                }
                next[i1] = Math.Tanh(sum);
            }
            (prev, next) = (next, prev);
        }
        Array.Resize(ref prev, _model.Last());
        return  prev;
    }
}