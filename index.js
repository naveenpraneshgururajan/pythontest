import React from "react";

export default MultiSelector = () => {
  const cityList = ["Chennai", "Mumbai", "Goa"];
  const [data, setData] = React.useState("");
  const handleOnChange = () => {
    setData((prev) => {
      if (prev.includes(city)) {
        return prev.filter((c) => c !== city);
      } else {
        return [...prev, city];
      }
    });
  };
  return (
    <>
      {cityList.map(() => {
        <select value={city} onChange={handleOnChange(city)}></select>;
      })}
    </>
  );
};
