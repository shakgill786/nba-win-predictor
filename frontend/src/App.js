import { useState } from "react";
import axios from "axios";

function App() {
  const [team, setTeam] = useState("");
  const [opp, setOpp] = useState("");
  const [homeAway, setHomeAway] = useState("home");
  const [prob, setProb] = useState(null);

  const handleSubmit = async e => {
    e.preventDefault();
    const res = await axios.post("/api/predict", {
      team, opponent: opp, home_away: homeAway
    });
    setProb(res.data.win_probability);
  };

  return (
    <div className="p-4">
      <h1>Game Outcome Predictor</h1>
      <form onSubmit={handleSubmit} className="space-y-2">
        <input
          placeholder="Your Team"
          value={team}
          onChange={e => setTeam(e.target.value)}
          required
        />
        <input
          placeholder="Opponent"
          value={opp}
          onChange={e => setOpp(e.target.value)}
          required
        />
        <select
          value={homeAway}
          onChange={e => setHomeAway(e.target.value)}
        >
          <option value="home">Home</option>
          <option value="away">Away</option>
        </select>
        <button type="submit">Predict</button>
      </form>

      {prob !== null && (
        <div className="mt-4">
          <strong>Win Probability:</strong> {(prob * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}

export default App;

