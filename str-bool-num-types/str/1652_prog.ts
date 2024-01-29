function maskCpf(value: string): string {
    return value
        .replace(/\D/g, '')
        .replace(/(\d{3})(\d)/, '$1.$2')
        .replace(/(\d{3})(\d)/, '$1.$2')
        .replace(/(\d{3})(\d{1,2})/, '$1-$2')
        .replace(/(-\d{2})\d+?$/, '$1');
}

function maskCnpj(value: string): string {
    return value
        .replace(/\D/g, '')
        .replace(/(\d{2})(\d)/, '$1.$2')
        .replace(/(\d{3})(\d)/, '$1.$2')
        .replace(/(\d{3})(\d)/, '$1/$2')
        .replace(/(\d{4})(\d{1,2})/, '$1-$2')
        .replace(/(-\d{2})\d+?$/, '$1');
}

function maskCep(value: <FILL>): string {
    return value
        .replace(/\D/g, '')
        .replace(/(\d{2})(\d)/, '$1.$2')
        .replace(/(\d{3})(\d{1,3})/, '$1-$2')
        .replace(/(-\d{3})\d+?$/, '$1');
}

function maskTelefone(value: string): string {
    return value
        .replace(/\D/g, '')
        .replace(/(\d{2})(\d)/, '($1) $2')
        .replace(/(\d{5})(\d{1,3})/, '$1-$2')
        .replace(/(-\d{4})\d+?$/, '$1');
}

function maskCurrency(value: string): string {
    value = value.replace(/\D/g, '');

    if ((value.trim().length >= 4) && (value.startsWith('0')))
        value = value.substring(1, value.trim().length);

    switch (value.trim().length) {
        case 0:
            return '0,00';
        case 1:
            return value.replace(/(\d{1})/, '0,0$1');
        case 2:
            return value.replace(/(\d{2})/, '0,$1');
        case 3:
            return value.replace(/(\d{1})(\d{2})/, '$1,$2');
        case 4:
            return value.replace(/(\d{2})(\d{2})/, '$1,$2');
        case 5:
            return value.replace(/(\d{3})(\d{2})/, '$1,$2');
        case 6:
            return value.replace(/(\d{1})(\d{3})(\d{2})/, '$1.$2,$3');
        case 7:
            return value.replace(/(\d{2})(\d{3})(\d{2})/, '$1.$2,$3');
        case 8:
            return value.replace(/(\d{3})(\d{3})(\d{2})/, '$1.$2,$3');
        case 9:
            return value.replace(/(\d{1})(\d{3})(\d{3})(\d{2})/, '$1.$2.$3,$4');
        case 10:
            return value.replace(/(\d{2})(\d{3})(\d{3})(\d{2})/, '$1.$2.$3,$4');
        case 11:
            return value.replace(/(\d{3})(\d{3})(\d{3})(\d{2})/, '$1.$2.$3,$4');
        default:
            return value;
    }
}

function maskNumber(value: string): string {
    let valor = value
        .replace(/\D/g, '')
        .replaceAll('.', '')
        .replaceAll(',', '');

    return (valor === '')
        ? '0'
        : String(parseInt(valor));
}

export type MaskType = 'cpf' | 'cnpj' | 'cep' | 'telefone' | 'currency' | 'number';

export function maskValue(value: string, mask: MaskType): string {
    switch (mask) {
        case 'cpf':
            return maskCpf(value);
        case 'cnpj':
            return maskCnpj(value);
        case 'cep':
            return maskCep(value);
        case 'telefone':
            return maskTelefone(value);
        case 'currency':
            return maskCurrency(value);
        case 'number':
            return maskNumber(value);
        default:
            return value;
    };
}