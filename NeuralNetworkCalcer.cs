using System.Numerics;
using System.Runtime.Intrinsics;

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
        var max = _model.Max();
        Span<float> prev = stackalloc float[max];
        new Span<float>(inputs, startInput, _model[0]).CopyTo(prev);
        Span<float> next = stackalloc float[max];
        var sinapsIndex = 0;
        var vectorLength = Vector<float>.Count;
        for (var i = 1; i < _model.Length; i++)
        {
            var prevLen = _model[i - 1];
            var len = _model[i];
            for (var i1 = 0; i1 < len; i1++)
            {
                var sum = 0f;
                var fromVector = prevLen / vectorLength;
                if (fromVector > 0)
                {
                    for (int j = 0; j < fromVector; j++)
                    {
                        var prevVector = new Vector<float>(prev.Slice(vectorLength * j, vectorLength));
                        var currentNeuronValues = new Vector<float>(new Span<float>(synapses, sinapsIndex, vectorLength));
                        var dot = Vector.Dot(prevVector, currentNeuronValues);
                        sum += dot;
                        sinapsIndex += vectorLength;
                    }
                }
                for (var i2 = fromVector * vectorLength; i2 < prevLen; i2++)
                {
                    var p = prev[i2];
                    var k = p * synapses[sinapsIndex];
                    sum += k;
                    sinapsIndex++;
                }
                next[i1] = MathF.Tanh(sum);
            }
            var tmp = prev;
            prev = next;
            next = tmp;
        }
        var result = new float[_model.Last()];
        prev.Slice(0, _model.Last()).CopyTo(result);
        return  result;
    }
}