using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Grammophone.CRF
{
	/// <summary>
	/// Exception during CRF training or running.
	/// </summary>
	[Serializable]
	public class CRFException : Exception
	{
		public CRFException(string message) : base(message) { }

		public CRFException(string message, Exception inner) : base(message, inner) { }

		/// <summary>
		/// Used for serialization.
		/// </summary>
		protected CRFException(
			System.Runtime.Serialization.SerializationInfo info,
			System.Runtime.Serialization.StreamingContext context)
			: base(info, context) { }
	}
}
