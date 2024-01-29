interface MixColorRes {
  light: Object,
  dark: Object
}

/**
 * color mix
 * @param color1 
 * @param color2 
 * @param weight 
 * @returns 
 */
 export function colorMix (color1: <FILL>, color2: string, weight: number) {
  weight = Math.max(Math.min(Number(weight), 1), 0)
  const r1 = parseInt(color1.substring(1, 3), 16)
  const g1 = parseInt(color1.substring(3, 5), 16)
  const b1 = parseInt(color1.substring(5, 7), 16)
  const r2 = parseInt(color2.substring(1, 3), 16)
  const g2 = parseInt(color2.substring(3, 5), 16)
  const b2 = parseInt(color2.substring(5, 7), 16)
  const r = Math.round(r1 * (1 - weight) + r2 * weight)
  const g = Math.round(g1 * (1 - weight) + g2 * weight)
  const b = Math.round(b1 * (1 - weight) + b2 * weight)
  const rr = ('0' + (r || 0).toString(16)).slice(-2)
  const gg = ('0' + (g || 0).toString(16)).slice(-2)
  const bb = ('0' + (b || 0).toString(16)).slice(-2)
  return '#' + rr + gg + bb;
}

/**
 * darker color
 * @param {*} color 
 * @param {*} mode 
 * @returns 
 */
export function getDarkerColor(color, mode='light') {
  const baseColor = mode === 'light' ? '#000000' : '#ffffff'
  return colorMix(color, baseColor, 0.2)
}

/**
 * color level
 * @param {*} color 
 * @param {*} weightArr step Array
 * @param {*} mode 
 */
export function lighterLevelColor(color: string, weightArr: Array<number>, mode='light') {
  const colors:Array<string> = []
  const mixColor = mode === 'light' ? '#ffffff' : '#141414'
  weightArr.forEach(weight => {
    colors.push(colorMix(color, mixColor, weight))
  });
  return colors
}

/**
 * generate main colors
 * @param {*} colors base颜色
 * @returns 
 */
export function generateMainColors(colors = {}){
  const prefix = '--el-color-'
  let types = {
    primary: { light: '#409eff', dark: '#409eff' },
    success: { light: '#67c23a', dark: '#59b259' },
    warning: { light: '#e6a23c', dark: '#d6a356' },
    danger: { light: '#f56c6c', dark: '#cc5e5e' },
    error: { light: '#f56c6c', dark: '#cc5e5e' },
    info: {light: '#909399', dark: '#606266' }
  }
  types = { ...types, ...colors }
  const res = {} as MixColorRes
  Object.keys(types).forEach(type=>{
    const item = types[type]
    // weight
    const weightArr = [0.3, 0.5, 0.7, 0.8, 0.9]
    const lightColor = lighterLevelColor(item.light, weightArr)
    const darkColor = lighterLevelColor(item.dark, weightArr, 'dark')
    // light-mode color
    lightColor.forEach((light, index)=>{
      res.light = { ...res.light, ...{ [prefix+type+'-light-'+weightArr[index]*10]: light } }
    })
    // dark-mode color
    darkColor.forEach((dark, index)=>{
      res.dark = { ...res.dark, ...{ [prefix+type+'-light-'+weightArr[index]*10]: dark } }
    })
    // add base color
    res.light[prefix+type] = item.light
    res.dark[prefix+type] = item.dark
    // add darker color
    res.light[prefix+type+'-dark-2'] = getDarkerColor(item.light)
    res.dark[prefix+type+'-dark-2'] = getDarkerColor(item.dark, 'dark') 
  })
  return res
}


/**
 * generate text colors
 */
export function generateTextColors(){
  const prefix = '--el-text-color-'
  const lightColor = {
    primary: '#303133',
    regular: '#606266',
    secondary: '#909399',
    placeholder: '#a8abb2',
    disabled: '#c0c4cc'
  }
  const res = {} as MixColorRes
  const colorKeys = Object.keys(lightColor)
  colorKeys.forEach(key=>{
    res.light = { ...res.light, ...{ [prefix+key]: lightColor[key] } }
  })
  const darkWeight = [0.05, 0.15, 0.35, 0.45, 0.60]
  const darkColor = lighterLevelColor('#f0f5ff', darkWeight, 'dark')
  darkColor.forEach((dark, index)=>{
    res.dark = { ...res.dark, ...{ [prefix+colorKeys[index]]: dark } }
  })
  return res
}

/**
 * generate border colors
 */
export function generateBorderColors(){
  const prefix = '--el-border-color'
  const lightColor = {
    'darker': '#cdd0d6',
    'dark': '#d4d7de',
    '': '#dcdfe6',
    'light': '#e4e7ed',
    'lighter': '#ebeef5',
    'extra-light': '#f2f6fc'
  }
  const res = {} as MixColorRes
  const colorKeys = Object.keys(lightColor)
  colorKeys.forEach(key=>{
    res.light = { ...res.light, ...{ [key?(prefix+'-'+key):prefix]: lightColor[key] } }
  })
  const darkWeight = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
  const darkColor = lighterLevelColor('#f5f8ff', darkWeight, 'dark')
  darkColor.forEach((dark, index)=>{
    res.dark = { ...res.dark, ...{ [colorKeys[index] ? (prefix+'-'+colorKeys[index]) : prefix]: dark } }
  })
  return res
}

/**
 * generate fill colors
 */
export function generateFillColors(){
  const prefix = '--el-fill-color'
  const lightColor = {
    'darker': '#e6e8eb',
    'dark': '#ebedf0',
    '': '#f0f2f5',
    'light': '#f5f7fa',
    'lighter': '#fafafa',
    'extra-light': '#fafcff',
    'blank': '#ffffff'
  }
  const res = {} as MixColorRes
  const colorKeys = Object.keys(lightColor)
  colorKeys.forEach(key=>{
    res.light = { ...res.light, ...{ [key?(prefix+'-'+key):prefix]: lightColor[key] } }
  })
  const darkWeight = [0.80, 0.84, 0.88, 0.92, 0.96, 0.98, 1]
  const darkColor = lighterLevelColor('#fafcff', darkWeight, 'dark')
  darkColor.forEach((dark, index)=>{
    res.dark = { ...res.dark, ...{ [colorKeys[index] ? (prefix+'-'+colorKeys[index]) : prefix]: dark } }
  })
  return res
}

/**
 * generate bg colors
 * @returns 
 */
export function generateBgColors(){
  return {
    light: {
      '--el-bg-color': '#ffffff',
      '--el-bg-page': '#ffffff',
      '--el-bg-overlay': '#ffffff'
    },
    dark: {
      '--el-bg-color': '#141414',
      '--el-bg-page': '#0a0a0a',
      '--el-bg-overlay': '#1d1e1f'
    }
  }
}

/**
 * generate mask  colors
 */
export function generateMaskColors(){
  return {
    light: {
      '--el-overlay-color': 'rgba(0, 0, 0, 0.8)',
      '--el-overlay-color-light': 'rgba(0, 0, 0, 0.7)',
      '--el-overlay-color-lighter': 'rgba(0, 0, 0, 0.5)',
      '--el-mask-color': 'rgba(255, 255, 255, 0.9)',
      '--el-mask-color-extra-light': 'rgba(255, 255, 255, 0.3)'
    },
    dark: {
      '--el-overlay-color': 'rgba(255, 255, 255, 0.8)',
      '--el-overlay-color-light': 'rgba(255, 255, 255, 0.7)',
      '--el-overlay-color-lighter': 'rgba(255, 255, 255, 0.5)',
      '--el-mask-color': 'rgba(0, 0, 0, 0.9)',
      '--el-mask-color-extra-light': 'rgba(0, 0, 0, 0.3)'
    }
  }
}

/**
 * generate BoxShadow Colors
 * @returns 
 */
export function generateBoxShadowColors() {
  return {
    light: {
      '--el-box-shadow': '0px 12px 32px 4px rgba(0, 0, 0, 0.04),0px 8px 20px rgba(0, 0, 0, 0.08)',
      '--el-box-shadow-light': '0px 0px 12px rgba(0, 0, 0, 0.12)',
      '--el-box-shadow-lighter': '0px 0px 6px rgba(0, 0, 0, 0.12)',
      '--el-box-shadow-dark': '0px 16px 48px 16px rgba(0, 0, 0, 0.08),0px 12px 32px rgba(0, 0, 0, 0.12),0px 8px 16px -8px rgba(0, 0, 0, 0.16)'
    },
    dark: {
      '--el-box-shadow': '0px 12px 32px 4px rgba(0, 0, 0, 0.36),0px 8px 20px rgba(0, 0, 0, 0.72)',
      '--el-box-shadow-light': '0px 0px 12px rgba(0, 0, 0, 0.72)',
      '--el-box-shadow-lighter': '0px 0px 6px rgba(0, 0, 0, 0.72)',
      '--el-box-shadow-dark': '0px 16px 48px 16px rgba(0, 0, 0, 0.72),0px 12px 32px #000000,0px 8px 16px -8px #000000'
    }
  }
}
