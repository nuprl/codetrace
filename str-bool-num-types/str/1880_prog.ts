export type Settings = {
  parserName: <FILL>;
  typeHints: boolean;
  mainContents: string;
  mainName: string;
  cliName: string;
  argsName: string;
};

export const defaultSettings = (): Settings => {
  return {
    parserName: 'parser',
    typeHints: true,
    mainContents: '# Contents of main',
    mainName: 'main',
    cliName: 'cli',
    argsName: 'args',
  };
};

export type Argument = {
  name: string;
  type: string;
  variableName: string;
  default: string;
  required: boolean;
};

export const newArgument = (
  name: string,
  type: string,
  variableName: string = '',
  defaultValue: string = '',
  required: boolean = false,
): Argument => {
  if (variableName === '') {
    variableName = name.replace(/-/g, '');
  }
  return { name, type, variableName, default: defaultValue, required };
};

function argumentToText(argument: Argument, parserName: string) {
  switch (argument.type) {
    case 'bool':
      return `${parserName}.add_argument("${argument.name}", action="store_${argument.default}")`;
    default:
      let defaultDisplay;

      if (argument.type === 'str') {
        defaultDisplay = `"${argument.default}"`;
      } else {
        defaultDisplay = argument.default;
      }

      if (argument.default === '') {
        return `${parserName}.add_argument("${argument.name}", type=${argument.type})`;
      } else {
        return `${parserName}.add_argument("${argument.name}", type=${argument.type}, default=${defaultDisplay})`;
      }
  }
}

const argumentToMainParams = (argument: Argument, typeHints: boolean) => {
  if (typeHints) {
    return `${argument.variableName}: ${argument.type}`;
  } else {
    return `${argument.variableName}`;
  }
};

export const argparseCode = (args: Argument[], settings: Settings = defaultSettings()) => {
  const parserName: string = settings.parserName;
  const argsName: string = settings.argsName;
  const mainParameters: string[] = args.map((arg) => argumentToMainParams(arg, settings.typeHints));
  const argumentsText: string[] = args.map((arg) => argumentToText(arg, parserName));
  const returnText: string[] = args.map((x) => {
    if (x.type === 'str') {
      return `"${x.variableName}": ${argsName}.${x.variableName},`;
    } else {
      return `"${x.variableName}": ${x.type}(${argsName}.${x.variableName}),`;
    }
  });

  const mainReturnType = settings.typeHints ? ' -> None' : '';
  const cliReturnType = settings.typeHints ? ' -> Dict[str, Any]' : '';
  const typeImports = settings.typeHints ? '\nfrom typing import Dict, Any' : '';

  const output = `import argparse${typeImports}


def ${settings.mainName}(${mainParameters.join(', ')})${mainReturnType}:
    ${settings.mainContents}
    return


def ${settings.cliName}()${cliReturnType}:
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    ${parserName} = argparse.ArgumentParser(formatter_class=formatter_class)

    ${argumentsText.join('\n    ')}

    ${argsName} = ${parserName}.parse_args()

    return {${returnText.join('\n            ').slice(0, -1)}}


if __name__ == '__main__':
    ${argsName} = ${settings.cliName}()
    ${settings.mainName}(${args.map((x) => `${x.variableName}=${argsName}["${x.variableName}"]`).join(',\n         ')})
`;
  return output;
};
