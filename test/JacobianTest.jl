module JacobianTest

using Base.Test
using ForwardDiff
using ForwardDiff: default_value, KWARG_DEFAULTS

########################
# @jacobian/@jacobian! #
########################

const ALLRESULTS_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :allresults))})
const CHUNK_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :chunk))})
const INPUT_MUTATES_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :input_mutates))})
const OUTPUT_MUTATES_DEFAULT = :(Val{$(default_value(KWARG_DEFAULTS, :output_mutates))})

@test macroexpand(:(ForwardDiff.@jacobian(sin))) == :(ForwardDiff.jacobian(sin, $ALLRESULTS_DEFAULT, $CHUNK_DEFAULT, $INPUT_MUTATES_DEFAULT, $OUTPUT_MUTATES_DEFAULT))
@test macroexpand(:(ForwardDiff.@jacobian(sin; output_mutates=1, allresults=2, input_mutates=3, chunk=4))) == :(ForwardDiff.jacobian(sin, Val{2}, Val{4}, Val{3}, Val{1}))
@test macroexpand(:(ForwardDiff.@jacobian(sin, chunk=1, output_mutates=2))) == :(ForwardDiff.jacobian(sin, $ALLRESULTS_DEFAULT, Val{1}, $INPUT_MUTATES_DEFAULT, Val{2}))

@test macroexpand(:(ForwardDiff.@jacobian(sin, x))) == :(ForwardDiff.jacobian(sin, x, $ALLRESULTS_DEFAULT, $CHUNK_DEFAULT))
@test macroexpand(:(ForwardDiff.@jacobian(sin, x, allresults=1, chunk=2))) == :(ForwardDiff.jacobian(sin, x, Val{1}, Val{2}))
@test macroexpand(:(ForwardDiff.@jacobian(sin, x; chunk=1))) == :(ForwardDiff.jacobian(sin, x, $ALLRESULTS_DEFAULT, Val{1}))
@test macroexpand(:(ForwardDiff.@jacobian(sin, x; allresults=1))) == :(ForwardDiff.jacobian(sin, x, Val{1}, $CHUNK_DEFAULT))

@test macroexpand(:(ForwardDiff.@jacobian!(sin, output, x))) == :(ForwardDiff.jacobian!(sin, output, x, $ALLRESULTS_DEFAULT, $CHUNK_DEFAULT))
@test macroexpand(:(ForwardDiff.@jacobian!(sin, output, x, allresults=1, chunk=2))) == :(ForwardDiff.jacobian!(sin, output, x, Val{1}, Val{2}))
@test macroexpand(:(ForwardDiff.@jacobian!(sin, output, x; chunk=1, allresults=2))) == :(ForwardDiff.jacobian!(sin, output, x, Val{2}, Val{1}))
@test macroexpand(:(ForwardDiff.@jacobian!(sin, output, x, chunk=1))) == :(ForwardDiff.jacobian!(sin, output, x, $ALLRESULTS_DEFAULT, Val{1}))
@test macroexpand(:(ForwardDiff.@jacobian!(sin, output, x; allresults=1))) == :(ForwardDiff.jacobian!(sin, output, x, Val{1}, $CHUNK_DEFAULT))

end # module
