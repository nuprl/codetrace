interface Color {
  red: number;
  green: number;
  blue: number;
}

export const colorGradient = (
  fadeFraction: number,
  rgbColor1: Color,
  rgbColor2: Color,
  rgbColor3?: Color
): Color => {
  let color1 = rgbColor1;
  let color2 = rgbColor2;
  let fade = fadeFraction;

  // Do we have 3 colors for the gradient? Need to adjust the params.
  if (rgbColor3) {
    fade = fade * 2;

    // Find which interval to use and adjust the fade percentage
    if (fade >= 1) {
      fade -= 1;
      color1 = rgbColor2;
      color2 = rgbColor3;
    }
  }

  const diffRed = color2.red - color1.red;
  const diffGreen = color2.green - color1.green;
  const diffBlue = color2.blue - color1.blue;

  const gradient = {
    red: parseInt(Math.floor(color1.red + diffRed * fade).toString(), 10),
    green: parseInt(Math.floor(color1.green + diffGreen * fade).toString(), 10),
    blue: parseInt(Math.floor(color1.blue + diffBlue * fade).toString(), 10),
  };

  // return 'rgb(' + gradient.red + ',' + gradient.green + ',' + gradient.blue + ')';
  return gradient;
};

export const perc2color = (
  perc: number,
  maxPerc: <FILL>,
  color1: Color,
  color2: Color,
  color3?: Color
): string => {
  perc = perc / maxPerc;
  if (color3) {
    perc = perc * 2;
    if (perc >= 1) {
      perc -= 1;
      color1 = color2;
      color2 = color3;
    }
  }
  const diffRed = color2.red - color1.red;
  const diffGreen = color2.green - color1.green;
  const diffBlue = color2.blue - color1.blue;
  const gradient = {
    red: parseInt(Math.floor(color1.red + diffRed * perc).toString(), 10),
    green: parseInt(Math.floor(color1.green + diffGreen * perc).toString(), 10),
    blue: parseInt(Math.floor(color1.blue + diffBlue * perc).toString(), 10),
  };
  const sum = 0x10000 * gradient.red + 0x100 * gradient.green + 0x1 * gradient.blue;
  return '#' + ('000000' + sum.toString(16)).slice(-6);
};
