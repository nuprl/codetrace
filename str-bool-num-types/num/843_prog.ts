type ColorPalette =
  | "PRIMARY"
  | "ACCENT"
  | "LINK"
  | "_2C2C2C"
  | "_FAFAFA"
  | "_F2F2F2"
  | "_EEEEEE"
  | "_18A0FB"
  | "_7FD1AE"
  | "_E0E0E0"
  | "_B00020";

const Color: { [k in ColorPalette]: string } = {
  // PRIMARY: "rgba(0, 180, 255, 1)",
  PRIMARY: "rgba(0, 0, 0, 1)",
  ACCENT: "rgba(255, 224, 0, 1)",
  LINK: "rgba(24, 160, 251, 1)",
  _2C2C2C: "rgba(44, 44, 44, 1)",
  _F2F2F2: "rgba(242, 242, 242, 1)",
  _FAFAFA: "rgba(250, 250, 250, 1)",
  _EEEEEE: "rgba(238, 238, 238, 1)",
  _18A0FB: "rgba(24, 160, 251, 1)",
  _7FD1AE: "rgba(127, 209, 174, 1)",
  _E0E0E0: "rgba(224, 224, 224, 1)",
  _B00020: "rgba(176, 0, 32, 1)",
};

export function getColor(opacity: number = 1) {
  return {
    TEXT: {
      ACCENT: `rgba(0, 153, 255, ${opacity})`,
      DANGER: `rgba(213, 0, 0, ${opacity})`,
      CONTRAST: `rgba(255, 255, 255, ${opacity})`,
      PRIMARY: `rgba(0, 0, 0, 0.87)`,
      SECONDARY: `rgba(0, 0, 0, 0.60)`,
      DISABLED: `rgba(0, 0, 0, 0.38)`,
    },
    BORDER: {
      LIGHT: `rgba(0,0,0,0.05)`,
      DARK: `rgba(0,0,0,0.2)`,
      ACCENT: `rgba(189, 229, 255, ${opacity})`,
      DANGER: `rgba(213, 0, 0, ${opacity})`,
      ONFOCUS: `rgba(0, 153, 255, ${opacity})`,
      ONHOVER: `rgba(0,0,0,0.20)`
    },
    BACKGROUND: {
      PRIMARY: `rgba(255, 255, 255, ${opacity})`,
      ACCENT: `rgba(0, 153, 255, ${opacity})`,
      ACCENT_LIGHT_1: `rgba(226, 243, 255, ${opacity})`,
      ACCENT_LIGHT_2: `rgba(245, 251, 255, ${opacity})`,
      PRIMARY_1: `rgba(250, 251, 252, ${opacity})`,
      ACCENT_LIGHT_3: `rgba(221, 227, 238, ${opacity})`,
      ERROR: `rgba(255, 224, 231, ${opacity})`,
    },
  };
}

export const getRGBA = (color: ColorPalette, opacity: <FILL> = 1) => {
  switch (color) {
    case "PRIMARY":
      return `rgba(0, 0, 0, ${opacity})`;
    // return `rgba(0, 118, 255, ${opacity})`;
    case "ACCENT":
      return `rgba(255, 224, 0, ${opacity})`;
    case "LINK":
      return `rgba(24, 160, 251, ${opacity})`;
    case "_2C2C2C":
      return `rgba(44, 44, 44, ${opacity})`;
    case "_FAFAFA":
      return `rgba(250, 250, 250, ${opacity})`;
    case "_F2F2F2":
      return `rgba(242, 242, 242, ${opacity})`;
    case "_EEEEEE":
      return `rgba(238, 238, 238, ${opacity})`;
    case "_18A0FB":
      return `rgba(24, 160, 251, ${opacity})`;
    case "_E0E0E0":
      return `rgba(224, 224, 224, ${opacity})`;
    case "_B00020":
      return `rgba(176, 0, 32, ${opacity})`;
    case "_7FD1AE":
      return `rgba(127, 209, 174, ${opacity})`;
    default:
      return "rgba(0,0,0,1)";
  }
};

// https://gist.github.com/Chak10/dc24c61c9bf2f651cb6d290eeef864c1
export function randDarkColor() {
  var lum = -0.25;
  var hex = String(
    "#" + Math.random().toString(16).slice(2, 8).toUpperCase()
  ).replace(/[^0-9a-f]/gi, "");
  if (hex.length < 6) {
    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  }
  var rgb = "#",
    c,
    i;
  for (i = 0; i < 3; i++) {
    c = parseInt(hex.substr(i * 2, 2), 16);
    c = Math.round(Math.min(Math.max(0, c + c * lum), 255)).toString(16);
    rgb += ("00" + c).substr(c.length);
  }
  return rgb;
}

export default Color;
