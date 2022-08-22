public class CalcRequest
{
    public int[] Model { get; set; }
    public string[] Input { get; set; }
    public double[] Synapses { get; set; }
}

public class CalcResponse
{
    public List<string> UnkownInputs { get; set; }
    public double[][] Outputs { get; set; }
}