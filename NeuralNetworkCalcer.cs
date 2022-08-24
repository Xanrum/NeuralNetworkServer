namespace NeuralNetworkServer;

public class NeuralNetworkCalcer
{
    public NeuralNetworkCalcer(int[] model)
    {
        _model = model;
    }
    private readonly int[] _model;
    

    public float[] Calc(float[] inputs, int startInput, float[] synapses)
    {
        
        var prev = new Span<float>(inputs, startInput, _model[0]).ToArray();
        var sinapsIndex = 0;
        for (var i = 1; i < _model.Length; i++)
        {
            var next = new float[_model[i]];
            var len = _model[i];
            for (var i1 = 0; i1 < len; i1++)
            {
                var sum = 0f;
                foreach (var p in prev)
                {
                    sum += p * synapses[sinapsIndex];
                    sinapsIndex++;
                }
                next[i1] = MathF.Tanh(sum);
            }
            prev = next;
        }
        return prev;
    }
}