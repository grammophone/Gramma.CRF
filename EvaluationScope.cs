namespace Gramma.CRF
{
	/// <summary>
	/// This scope is hint during the creation of a <see cref="LinearChainCRF{I, T}.FeatureFunctionsProvider"/>
	/// indicating the occasion where evaluation takes place.
	/// Typically, the scope is a hint for caching precomputed structures required by feature functions.
	/// </summary>
	public enum EvaluationScope
	{
		/// <summary>
		/// Hint that evaluation takes place in training phase.
		/// </summary>
		Training,

		/// <summary>
		/// Hint that evaluation takes place at a completely trained Conditional Random Field.
		/// </summary>
		Running
	}
}