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
        Span<double> prev = stackalloc double[max];
        new Span<double>(inputs, startInput, _model[0]).CopyTo(prev);
        Span<double> next = stackalloc double[max];
        var sinapsIndex = 0;
        for (var i = 1; i < _model.Length; i++)
        {
            var len = _model[i];
            for (var i1 = 0; i1 < len; i1++)
            {
                var sum = 0d;
                for (var i2 = 0; i2 < _model[i-1]; i2++)
                {
                    var p = prev[i2];
                    sum += p * synapses[sinapsIndex];
                    sinapsIndex++;
                }
                next[i1] = Math.Tanh(sum);
            }
            var tmp = prev;
            prev = next;
            next = tmp;
        }
        var result = new double[_model.Last()];
        prev.Slice(0, _model.Last()).CopyTo(result);
        return  result;
    }
}