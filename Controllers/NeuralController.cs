using Microsoft.AspNetCore.Mvc;
using System.Collections.Concurrent;

namespace NeuralNetworkServer.Controllers;

[ApiController]
[Route("[controller]")]
public class NeuralController : ControllerBase
{

    private static readonly ConcurrentDictionary<string, double[]> InputsCache = new();

    [HttpPost(Name = "Load")]
    public void Load(LoadInputsRequest request)
    {
        foreach (var item in request.Data)
        {
            InputsCache[item.Key] = item.Inputs;
        }
    }
    
    [HttpPost(Name = "Calc")]
    public CalcResponse Calc(CalcRequest request)
    {
        var inputs = new List<double[]>();
        var unknownInputs = new List<string>();
        foreach (var s in request.Input)
        {
            if (InputsCache.TryGetValue(s, out var input))
            {
                inputs.Add(input);
            }
            else
            {
                unknownInputs.Add(s);
            }
        }
        if (unknownInputs.Count != 0)
            return new()
            {
                UnkownInputs = unknownInputs,
                Outputs = Array.Empty<double[]>()
            };
        var calcer = new NeuralNetworkCalcer(request.Model);
        var outputs = new double[inputs.Count][];
        for (var i = 0; i < inputs.Count; i++)
        {
            outputs[i] = calcer.Calc(inputs[i], request.Synapses);
        }
        return new()
        {
            UnkownInputs = unknownInputs,
            Outputs = outputs
        };
    }
}