verdict=easy

Note I removed the docstr that gave away types.

In "void" I further deleted bottom half of prog that gave away answer. Otherwise the LLM perform great (due to copying heads)
EDIT: even with bottom deleted LLM return `Promise(<void>)` which is equiv to `void` here